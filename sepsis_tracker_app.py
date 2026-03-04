import streamlit as st
import requests
import json
import google.generativeai as genai
import datetime
import time
import pandas as pd
import re
import io

# 尝试导入 PDF 解析库，并做友好提示
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# ==========================================
# 1. 页面配置与状态初始化
# ==========================================
st.set_page_config(page_title="Pro Literature Tracker", layout="wide")
st.title("🩸 Pro Pathogen RNA Tracker")
st.markdown("万能语法翻译引擎 + 本地 PDF 精读 + Sepsis 项目专属顾问 🚀")

if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'ai_analyses' not in st.session_state:
    st.session_state.ai_analyses = {}
if 'cn_summaries' not in st.session_state:
    st.session_state.cn_summaries = {}
if 'local_analysis' not in st.session_state:
    st.session_state.local_analysis = None

# ==========================================
# 2. LLM 提示词 (包含专属项目顾问)
# ==========================================
SYSTEM_PROMPT = """
You are a top-tier molecular biology literature analysis engine.
Read the provided scientific abstract and extract the following key parameters.
Strictly return a valid JSON object. Do not include markdown formatting. If a detail is missing, output "Not specified in abstract".

{
  "research_purpose": "Purpose and significance of the study",
  "target_pathogens": "Specific targets (e.g., Bacteria, Fungi, specific species)",
  "sample_type": "Type of sample (e.g., Whole blood, serum, plasma)",
  "sample_prep_and_extraction": "Methods for sample preparation and nucleic acid extraction",
  "experimental_methods": "Core methodologies (e.g., RT-qPCR, 16S/18S/ITS sequencing)",
  "primer_probe_sequences": "Specific primer/probe sequences if mentioned",
  "main_results": "Key findings, including LOD, sensitivity, or specificity",
  "limitations": "Study limitations if mentioned"
}
"""

CN_SUMMARY_PROMPT = """
请作为专业的分子生物学科研助手，对以下这篇英文摘要进行全面且深度的中文解析。
请严格包含以下两部分内容，并使用 Markdown 排版：

### 📝 【摘要全译】
（请将原英文摘要进行准确、流畅、且符合医学专业术语的完整中文翻译。不要省略任何研究背景和结论信息。）

### 🎯 【核心数据提炼】
（请用简明扼要的列表形式，直接列出这篇文献最核心的实验参数）
- **研究样本**：(说明具体样本类型，如全血、血清，及体积等，若无则写未提及)
- **提取与检测方法**：(说明提取策略、试剂盒、核心扩增技术等)
- **关键结果与参数**：(说明灵敏度 LOD、特异性、检测时间等具体数值)
"""

PROJECT_CONTEXT = """
【我的 Sepsis 快速检测项目背景】：
基于 10mL 全血样品，进行超灵敏度的、以每种菌特异性 16S rRNA 片段（包括真菌靶标）为靶标的超快速检测。
我的核心三步法与痛点如下：
1. **样本前处理**：10mL 全血的血细胞裂解与清洗。目标是获得不破碎的细菌，极力提高极低浓度下的细菌回收率。
2. **核酸提取**：对清洗后的细菌采用“超声”物理方式进行核酸释放，获取 total DNA 和 RNA。
3. **特异性检测**：设计各菌种 16S rRNA 特异性序列。建立高灵敏 RT-qPCR。最大痛点是必须避免/消除 16S 的试剂本底信号（NC 起线/假阳性污染）。
"""

LOCAL_PAPER_PROMPT = f"""
请作为顶尖的分子诊断与微生物学专家，仔细阅读以下我提供的文献全文（或文本片段），并输出一份深度的中文评估报告。

请严格按照以下 Markdown 格式输出：

### 📝 【文献中文精粹】
（提炼本文的主要研究目的、核心方法、最重要的数据结果和最终结论）

### 🧪 【核心技术参数提取】
- **样本处理**：（记录其处理的全血体积、裂解方式、富集方法等）
- **核酸提取**：（提取试剂、物理/化学打断方式等）
- **检测方法与靶标**：（PCR/RT-qPCR等，是否提供具体引物/探针序列信息，LOD 灵敏度等）

### 💡 【对您 Sepsis 项目的深度对比与评估】
已知您的项目背景如下：
{PROJECT_CONTEXT}

请结合这篇文献的内容，针对您的三个关键环节进行逐一对比、启发评估。如果文献中没有相关内容，请说明“本文未涉及”，并给出您作为顶尖专家的额外分析建议：
1. **关于 10mL 全血裂解与细菌富集**：（文献的方法对比您的思路有何优劣？能否帮助您提高细菌回收率？）
2. **关于超声释放核酸 (DNA/RNA)**：（文献的提取方式有何不同？您的超声方案有何潜在风险或优化空间？）
3. **关于 16S 特异性扩增与本底消除**：（文献是如何解决背景污染/NC起线问题的？对您的特异性靶标设计有何借鉴？）

### 🛠️ 【下一步实验优化建议】
（基于您的痛点和这篇文献的启发，给出 2-3 条切实可行的实验操作改进建议）
"""

# ==========================================
# 3. 跨平台数据库引擎模块
# ==========================================
def search_pubmed(query, years_back, max_results):
    current_year = datetime.datetime.now().year
    mindate = current_year - years_back
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed", "term": query, "retmode": "json", "retmax": max_results,
        "mindate": f"{mindate}/01/01", "maxdate": f"{current_year}/12/31",
        "datetype": "pdat", "sort": "date"
    }
    id_list = requests.get(search_url, params=search_params).json().get("esearchresult", {}).get("idlist", [])
    if not id_list: return []
    
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    summary_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "json"}
    result_dict = requests.get(summary_url, params=summary_params).json().get("result", {})
    
    papers = []
    for pmid in id_list:
        if pmid in result_dict:
            papers.append({
                "id": pmid,
                "title": result_dict[pmid].get("title", "Unknown Title"),
                "pubdate": result_dict[pmid].get("pubdate", "Unknown Date"),
                "source": result_dict[pmid].get("source", "PubMed Journal"),
                "abstract": "",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
    return papers

def search_epmc(query, years_back, max_results):
    current_year = datetime.datetime.now().year
    mindate = current_year - years_back
    epmc_query = f'({query}) AND FIRST_PDATE:[{mindate} TO {current_year}]'
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": epmc_query, "format": "json", "resultType": "core", "pageSize": max_results}
    results = requests.get(url, params=params).json().get("resultList", {}).get("result", [])
    
    papers = []
    for item in results:
        paper_id = item.get("pmid") or item.get("doi") or item.get("id", "Unknown")
        if item.get("doi"): link = f"https://doi.org/{item.get('doi')}"
        elif item.get("pmcid"): link = f"https://europepmc.org/article/PMC/{item.get('pmcid')}"
        elif item.get("pmid"): link = f"https://pubmed.ncbi.nlm.nih.gov/{item.get('pmid')}/"
        else: link = f"https://europepmc.org/search?query={paper_id}"

        papers.append({
            "id": paper_id,
            "title": item.get("title", "Unknown Title"),
            "pubdate": item.get("firstPublicationDate", "Unknown Date"),
            "source": item.get("journalTitle", "Preprint / EPMC"),
            "abstract": item.get("abstractText", ""),
            "url": link 
        })
    return papers

def search_semantic_scholar(query, years_back, max_results):
    current_year = datetime.datetime.now().year
    mindate = current_year - years_back
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query[:100], # Semantic Scholar 仅接受处理后的干净短字符串
        "year": f"{mindate}-{current_year}",
        "limit": max_results,
        "fields": "paperId,title,year,venue,abstract,url,externalIds"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200: return []
        results = response.json().get("data", [])
        
        papers = []
        for item in results:
            if not item.get("abstract"): continue 
            p_id = item.get("paperId", "Unknown")
            ext_ids = item.get("externalIds", {})
            if "DOI" in ext_ids: link = f"https://doi.org/{ext_ids['DOI']}"
            elif "PubMed" in ext_ids: link = f"https://pubmed.ncbi.nlm.nih.gov/{ext_ids['PubMed']}/"
            else: link = item.get("url", f"https://www.semanticscholar.org/paper/{p_id}")
            papers.append({
                "id": p_id, "title": item.get("title", "Unknown Title"), "pubdate": str(item.get("year", "Unknown Date")),
                "source": item.get("venue", "Semantic Scholar"), "abstract": item.get("abstract", ""), "url": link
            })
        return papers
    except Exception as e:
        st.error(f"Semantic Scholar 请求失败: {e}")
        return []

def fetch_pubmed_abstract(pmid):
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {"db": "pubmed", "id": pmid, "retmode": "text", "rettype": "abstract"}
    return requests.get(fetch_url, params=fetch_params).text

# ==========================================
# 4. AI 分析模块 (Gemini 2.5 Pro)
# ==========================================
def analyze_with_gemini_json(text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    full_prompt = f"{SYSTEM_PROMPT}\n\nAbstract Text to Analyze:\n{text}"
    for attempt in range(3):
        try:
            response = model.generate_content(full_prompt)
            clean_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            if "429" in str(e).lower() or "quota" in str(e).lower():
                if attempt < 2: time.sleep(3); continue
            raise e

def generate_quick_cn_summary(text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    full_prompt = f"{CN_SUMMARY_PROMPT}\n\nAbstract:\n{text}"
    for attempt in range(3):
        try:
            return model.generate_content(full_prompt).text.strip()
        except Exception as e:
            if "429" in str(e).lower() or "quota" in str(e).lower():
                if attempt < 2: time.sleep(3); continue
            raise e

def analyze_local_paper(text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    full_prompt = f"{LOCAL_PAPER_PROMPT}\n\n======================\n【提供的文献内容】：\n{text}"
    for attempt in range(3):
        try:
            return model.generate_content(full_prompt).text.strip()
        except Exception as e:
            if "429" in str(e).lower() or "quota" in str(e).lower():
                if attempt < 2: time.sleep(3); continue
            raise e

def convert_to_csv():
    export_data = []
    for paper in st.session_state.search_results:
        p_id = paper['id']
        if p_id in st.session_state.ai_analyses or p_id in st.session_state.cn_summaries:
            row = {
                "文献 ID": p_id, "发表时间": paper['pubdate'], "期刊/来源": paper['source'],
                "文献标题": paper['title'], "直达链接": paper['url'],
                "🇨🇳 中文全译与解析": st.session_state.cn_summaries.get(p_id, "未生成")
            }
            ai_data = st.session_state.ai_analyses.get(p_id, {})
            row["研究目的"] = ai_data.get("research_purpose", "未生成")
            row["目标病原体"] = ai_data.get("target_pathogens", "未生成")
            row["样本类型"] = ai_data.get("sample_type", "未生成")
            row["核酸提取与样本制备"] = ai_data.get("sample_prep_and_extraction", "未生成")
            row["核心实验方法"] = ai_data.get("experimental_methods", "未生成")
            row["引物/探针序列"] = ai_data.get("primer_probe_sequences", "未生成")
            row["主要结果与灵敏度"] = ai_data.get("main_results", "未生成")
            row["研究局限性"] = ai_data.get("limitations", "未生成")
            export_data.append(row)
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False).encode('utf-8-sig')

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages: text += page.extract_text() + "\n"
    return text

# ==========================================
# 🌟 核心引擎：万能跨平台语法翻译器
# ==========================================
def translate_cross_db_query(base_query, db_type):
    """不论你输入什么 PubMed 语法，自动为你翻译成对应数据库的母语"""
    if "PubMed" in db_type:
        # PubMed 原生支持 [Title]，直接通行
        return base_query
        
    elif "Europe" in db_type:
        # 把 PubMed 语法翻译为 Europe PMC 语法：[Title] -> TITLE:
        # 匹配任何带引号或不带引号的单词后面跟着 [Title] 的情况
        q = re.sub(r'(\"[^\"]+\"|[\w\-]+)\[Title\]', r'TITLE:\1', base_query, flags=re.IGNORECASE)
        q = re.sub(r'(\"[^\"]+\"|[\w\-]+)\[Title/Abstract\]', r'(TITLE:\1 OR ABSTRACT:\1)', q, flags=re.IGNORECASE)
        q = re.sub(r'(\"[^\"]+\"|[\w\-]+)\[Mesh\]', r'\1', q, flags=re.IGNORECASE)
        return q
        
    else:
        # 翻译给 Semantic Scholar (AI 语义引擎不认标点，全部扒光)
        q = re.sub(r'\[.*?\]', '', base_query) # 删去所有标签
        q = re.sub(r'[\(\)\"]', ' ', q) # 删去括号和引号
        q = q.replace(' OR ', ' ').replace(' AND ', ' ').replace(' NOT ', ' ')
        q = re.sub(r'\s+', ' ', q).strip()
        return q

# ==========================================
# 6. UI 与交互逻辑
# ==========================================
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

st.sidebar.header("🔍 Advanced Search Builder")
db_choice = st.sidebar.radio(
    "📚 Select Database", 
    ["PubMed (Standard, highly precise)", "Europe PMC (Includes Preprints)", "Semantic Scholar (AI-driven, Broadest)"]
)

search_mode = st.sidebar.radio("🎯 检索模式", ["🧩 引导式拼接模式 (自动包裹标签)", "💻 专家原码模式 (输入PubMed原生代码)"])

# 1. 首先构建基于 PubMed 语法的“基准逻辑词 (Base Query)”
if search_mode == "🧩 引导式拼接模式 (自动包裹标签)":
    st.sidebar.markdown("<small>提示：务必将你要组合的词组用 **双引号** 包裹，如 `\"RT-qPCR\"`</small>", unsafe_allow_html=True)
    title_kw = st.sidebar.text_area("文章【标题】必须包含:", value='"sepsis" AND "RT-qPCR"', height=60)
    mesh_kw = st.sidebar.text_input("医学主题词 MeSH (仅PubMed):", value="")
    abs_kw = st.sidebar.text_area("文章【摘要/全文】必须包含:", value='"whole blood"', height=60)
    
    parts = []
    if title_kw:
        if '"' in title_kw:
            # 彻底修复：将 "RT-qPCR" 精确替换为 "RT-qPCR"[Title]
            t_kw = re.sub(r'\"([^\"]+)\"', r'"\1"[Title]', title_kw)
            parts.append(f"({t_kw})")
        else:
            parts.append(f"({title_kw})[Title]")
    if mesh_kw: parts.append(f'("{mesh_kw}"[Mesh])')
    if abs_kw:
        if '"' in abs_kw:
            a_kw = re.sub(r'\"([^\"]+)\"', r'"\1"[Title/Abstract]', abs_kw)
            parts.append(f"({a_kw})")
        else:
            parts.append(f"({abs_kw})[Title/Abstract]")
            
    base_query = " AND ".join(parts)

else:
    st.sidebar.markdown("<small>提示：直接从 PubMed 网页版复制检索式进来即可。系统会自动为您翻译适配到欧洲库等其他引擎！</small>", unsafe_allow_html=True)
    base_query = st.sidebar.text_area(
        "📝 输入 PubMed 原生检索式:", 
        value='("sepsis"[Title] OR "bloodstream infection"[Title]) AND ("RT-qPCR"[Title/Abstract]) AND ("whole blood"[Title/Abstract])', 
        height=150
    )

# 2. 🌟 核心召唤：执行万能翻译！将 PubMed 语法翻译为你当前选中数据库的专属语法！
final_api_query = translate_cross_db_query(base_query, db_choice)

years_back = st.sidebar.selectbox("Time Range", [1, 3, 5, 10], index=2, format_func=lambda x: f"Last {x} Years")
max_results = st.sidebar.slider("Max papers to fetch", 10, 100, 50)

with st.sidebar.expander("👀 透明引擎：查看底层翻译过程", expanded=True):
    st.markdown("**您的 PubMed 基准逻辑:**")
    st.code(base_query, language="text")
    st.markdown(f"**🤖 AI 翻译为 `{db_choice.split()[0]}` 专属真实代码:**")
    st.code(final_api_query, language="text")

if st.sidebar.button("1. Fetch Summary List"):
    if not final_api_query.strip():
        st.sidebar.error("检索式不能为空！")
    else:
        with st.spinner(f"Searching {db_choice} with translated query..."):
            if "PubMed" in db_choice: st.session_state.search_results = search_pubmed(final_api_query, years_back, max_results)
            elif "Europe" in db_choice: st.session_state.search_results = search_epmc(final_api_query, years_back, max_results)
            else: st.session_state.search_results = search_semantic_scholar(final_api_query, years_back, max_results)
                
            if st.session_state.search_results:
                st.sidebar.success(f"Found {len(st.session_state.search_results)} highly relevant papers!")
            else:
                st.sidebar.warning("No papers found. 条件太苛刻或语法错误，请检查底部真实发送的代码！")

# 导出按钮区
if st.session_state.ai_analyses or st.session_state.cn_summaries:
    st.sidebar.markdown("---")
    st.sidebar.header("💾 导出科研成果")
    csv_bytes = convert_to_csv()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    st.sidebar.download_button("📥 下载已分析数据 (Excel格式)", data=csv_bytes, file_name=f"Pro_Analysis_Data_{current_date}.csv", mime="text/csv")

# ==========================================
# 🌟 主界面：双重 Tab 布局
# ==========================================
tab_search, tab_upload = st.tabs(["🌐 1. 在线文献检索与批量分析", "📂 2. 本地全文深度精读与专属 Sepsis 评估"])

with tab_search:
    if st.session_state.search_results:
        st.subheader(f"📑 Search Results from {db_choice}")
        st.caption("Sorted by publication date (Newest first). Click expander to view AI options.")
        
        for paper in st.session_state.search_results:
            p_id = paper['id']
            with st.expander(f"📅 {paper['pubdate']} | {paper['title']}"):
                st.markdown(f"**Source:** *{paper['source']}* | [🔗 访问原文 (View Full Article)]({paper['url']})")
                
                col1, col2 = st.columns(2)
                with col1:
                    if p_id not in st.session_state.cn_summaries:
                        if st.button("🇨🇳 深度翻译与核心提炼", key=f"cn_{p_id}"):
                            if not api_key: st.error("请在左侧输入 API Key。")
                            else:
                                with st.spinner("Pro 大脑正在逐句精译并提炼核心数据..."):
                                    try:
                                        abstract = paper['abstract'] if paper['abstract'] else fetch_pubmed_abstract(p_id)
                                        st.session_state.cn_summaries[p_id] = generate_quick_cn_summary(abstract, api_key)
                                        st.rerun()
                                    except Exception as e: st.error(f"❌ API 请求失败 ({str(e)})")
                    else:
                        st.success(f"💡 **中文精读报告**：\n{st.session_state.cn_summaries[p_id]}")

                with col2:
                    if p_id not in st.session_state.ai_analyses:
                        if st.button("🔬 生成深度结构化解析 (JSON)", key=f"deep_{p_id}"):
                            if not api_key: st.error("请在左侧输入 API Key。")
                            else:
                                with st.spinner("Pro 大脑正在提取实验参数..."):
                                    try:
                                        abstract = paper['abstract'] if paper['abstract'] else fetch_pubmed_abstract(p_id)
                                        res = analyze_with_gemini_json(abstract, api_key)
                                        res['abstract_text'] = abstract
                                        st.session_state.ai_analyses[p_id] = res
                                        st.rerun()
                                    except Exception as e: st.error(f"❌ API 请求失败 ({str(e)})")
                                
                if p_id in st.session_state.ai_analyses:
                    res = st.session_state.ai_analyses[p_id]
                    st.divider()
                    t1, t2, t3, t4 = st.tabs(["🎯 目的与病原体", "🧪 方法与制备", "📊 结果与局限", "📝 原文摘要"])
                    with t1:
                        st.markdown(f"**Research Purpose:** {res.get('research_purpose')}")
                        st.markdown(f"**Target Pathogens:** {res.get('target_pathogens')}")
                    with t2:
                        st.markdown(f"**Sample Type:** {res.get('sample_type')}")
                        st.markdown(f"**Sample Prep & Extraction:** {res.get('sample_prep_and_extraction')}")
                        st.markdown(f"**Experimental Methods:** {res.get('experimental_methods')}")
                        st.markdown(f"**Primer/Probe Sequences:** {res.get('primer_probe_sequences')}")
                    with t3:
                        st.markdown(f"**Main Results:** {res.get('main_results')}")
                        st.markdown(f"**Limitations:** {res.get('limitations')}")
                    with t4:
                        st.write(res.get('abstract_text'))
    else:
        st.info("👈 设置侧边栏条件并点击 'Fetch Summary List' 开始检索在线数据库。")

# --- 本地文献解析专区 ---
with tab_upload:
    st.markdown("### 📄 上传文献进行全文深度评估")
    st.info("💡 **专家提示**：Gemini 2.5 Pro 支持超大上下文。直接上传论文 PDF 全文，AI 将结合您专属的 10mL 全血超声提取与 16S 靶标项目进行深度对比与优化评估。")
    
    if not PYPDF_AVAILABLE:
        st.error("🚨 缺少 PDF 解析插件！请在终端运行 `pip install pypdf` 或将其加入您的 GitHub requirements.txt 中。")
    else:
        uploaded_file = st.file_uploader("📂 请上传一篇文献全文 (仅支持 .pdf 或 .txt)", type=['pdf', 'txt'])
        
        if uploaded_file is not None:
            if st.button("🔬 针对我的 Sepsis 项目进行深度诊断评估", type="primary"):
                if not api_key:
                    st.error("请先在左侧侧边栏输入您的 API Key。")
                else:
                    with st.spinner("🧠 正在阅读全文，并结合您的 3步法痛点进行对比思考，请稍候..."):
                        try:
                            if uploaded_file.name.endswith('.pdf'): paper_text = extract_text_from_pdf(uploaded_file)
                            else: paper_text = uploaded_file.getvalue().decode('utf-8')
                            
                            if len(paper_text) > 300000:
                                st.warning("文献过长，已截取前 300,000 字进行解析。")
                                paper_text = paper_text[:300000]
                            
                            st.session_state.local_analysis = analyze_local_paper(paper_text, api_key)
                        except Exception as e:
                            st.error(f"❌ 解析失败，请检查文件格式或 API 状态：{e}")
                            
            if st.session_state.local_analysis:
                st.success("✅ 解析完成！以下是为您量身定制的诊断评估报告：")
                st.markdown(st.session_state.local_analysis)