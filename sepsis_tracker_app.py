import streamlit as st
import requests
import json
import google.generativeai as genai
import datetime
import time
import pandas as pd
import re

# ==========================================
# 1. 页面配置与状态初始化
# ==========================================
st.set_page_config(page_title="Pro Literature Tracker", layout="wide")
st.title("🩸 Pro Pathogen RNA Tracker")
st.markdown("原生检索指令直通车 + 深度摘要全译解析 + 自动导出 🚀")

if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'ai_analyses' not in st.session_state:
    st.session_state.ai_analyses = {}
if 'cn_summaries' not in st.session_state:
    st.session_state.cn_summaries = {}

# ==========================================
# 2. LLM 提示词 (精读翻译 Prompt)
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

# ==========================================
# 3. 数据库引擎模块 
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
    
    clean_query = re.sub(r'\[.*?\]', '', query)
    clean_query = re.sub(r'[\(\)\"TITLE:ABSTRACT:]', ' ', clean_query)
    clean_query = clean_query.replace(' OR ', ' ').replace(' AND ', ' ').replace(' NOT ', ' ')
    clean_query = re.sub(r'\s+', ' ', clean_query).strip()
    
    if len(clean_query) > 100: clean_query = clean_query[:100]
        
    params = {
        "query": clean_query,
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
                "id": p_id,
                "title": item.get("title", "Unknown Title"),
                "pubdate": str(item.get("year", "Unknown Date")),
                "source": item.get("venue", "Semantic Scholar"),
                "abstract": item.get("abstract", ""),
                "url": link
            })
        return papers
    except Exception as e:
        st.error(f"Semantic Scholar 引擎请求失败: {e}")
        return []

def fetch_pubmed_abstract(pmid):
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {"db": "pubmed", "id": pmid, "retmode": "text", "rettype": "abstract"}
    return requests.get(fetch_url, params=fetch_params).text

# ==========================================
# 4. AI 分析模块 (Gemini 2.5 Pro) - 已修复截断问题
# ==========================================
def analyze_with_gemini_json(text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    full_prompt = f"{SYSTEM_PROMPT}\n\nAbstract Text to Analyze:\n{text}"
    for attempt in range(3):
        try:
            response = model.generate_content(full_prompt)
            clean_text = response.text.strip()
            # 使用更安全的字符串替换方式，防止页面框架意外截断代码
            clean_text = clean_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            if "429" in str(e).lower() or "quota" in str(e).lower():
                if attempt < 2:
                    time.sleep(3) 
                    continue
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
                if attempt < 2:
                    time.sleep(3) 
                    continue
            raise e

# ==========================================
# 5. 数据导出引擎
# ==========================================
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

# ==========================================
# 6. UI 与交互逻辑 (🌟 全新双模式检索架构)
# ==========================================
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

st.sidebar.header("🔍 Advanced Search Builder")
db_choice = st.sidebar.radio(
    "📚 Select Database", 
    ["PubMed (Standard, highly precise)", "Europe PMC (Includes Preprints)", "Semantic Scholar (AI-driven, Broadest)"]
)

search_mode = st.sidebar.radio("🎯 检索模式", ["🧩 引导式拼接模式 (原生防呆)", "💻 专家原码模式 (100% 控制权)"])

if search_mode == "🧩 引导式拼接模式 (原生防呆)":
    st.sidebar.markdown("<small>提示：支持输入括号和 AND/OR，系统将为您安全添加字段标签。</small>", unsafe_allow_html=True)
    title_kw = st.sidebar.text_area("文章【标题】必须包含:", value='"sepsis" AND "RT-qPCR"', height=60)
    mesh_kw = st.sidebar.text_input("医学主题词 MeSH (仅PubMed):", value="")
    abs_kw = st.sidebar.text_area("文章【摘要/全文】必须包含:", value='"whole blood"', height=60)
    
    def build_smart_query():
        parts = []
        if "PubMed" in db_choice:
            if title_kw:
                # 智能微操：自动把 [Title] 挂载到每一个双引号内的词汇上
                if '"' in title_kw:
                    t_kw = re.sub(r'\"([^\"]+)\"', r'"\1"[Title]', title_kw)
                    parts.append(f"({t_kw})")
                else:
                    parts.append(f"({title_kw})[Title]")
            if mesh_kw: 
                parts.append(f'("{mesh_kw}"[Mesh])')
            if abs_kw:
                if '"' in abs_kw:
                    a_kw = re.sub(r'\"([^\"]+)\"', r'"\1"[Title/Abstract]', abs_kw)
                    parts.append(f"({a_kw})")
                else:
                    parts.append(f"({abs_kw})[Title/Abstract]")
        elif "Europe" in db_choice:
            if title_kw:
                if '"' in title_kw:
                    t_kw = re.sub(r'\"([^\"]+)\"', r'TITLE:"\1"', title_kw)
                    parts.append(f"({t_kw})")
                else:
                    parts.append(f'TITLE:({title_kw})')
            if mesh_kw: 
                parts.append(f'({mesh_kw})')
            if abs_kw:
                if '"' in abs_kw:
                    a_kw = re.sub(r'\"([^\"]+)\"', r'ABSTRACT:"\1"', abs_kw)
                    parts.append(f"({a_kw})")
                else:
                    parts.append(f'ABSTRACT:({abs_kw})')
        else:
            # Semantic Scholar 保持原生，不加任何干扰标签
            if title_kw: parts.append(f"{title_kw}")
            if abs_kw: parts.append(f"{abs_kw}")
        return " AND ".join(parts)
        
    final_query = build_smart_query()

else:
    st.sidebar.markdown("<small>提示：直接输入完整的底层检索式，系统不做任何修改直接发送！</small>", unsafe_allow_html=True)
    final_query = st.sidebar.text_area(
        "📝 输入原生检索式:", 
        value='("sepsis"[Title] OR "bloodstream infection"[Title]) AND ("RT-qPCR"[Title/Abstract]) AND ("whole blood"[Title/Abstract])', 
        height=150
    )

years_back = st.sidebar.selectbox("Time Range", [1, 3, 5, 10], index=2, format_func=lambda x: f"Last {x} Years")
max_results = st.sidebar.slider("Max papers to fetch", 10, 100, 50)

with st.sidebar.expander("👀 查看即将发送至服务器的真实代码", expanded=True):
    st.code(final_query, language="text")

# ==========================================
# 🚀 触发搜索与展示逻辑
# ==========================================
if st.sidebar.button("1. Fetch Summary List"):
    if not final_query.strip():
        st.sidebar.error("检索式不能为空！")
    else:
        with st.spinner(f"Searching {db_choice} with advanced filters..."):
            if "PubMed" in db_choice:
                st.session_state.search_results = search_pubmed(final_query, years_back, max_results)
            elif "Europe" in db_choice:
                st.session_state.search_results = search_epmc(final_query, years_back, max_results)
            else:
                st.session_state.search_results = search_semantic_scholar(final_query, years_back, max_results)
                
            if st.session_state.search_results:
                st.sidebar.success(f"Found {len(st.session_state.search_results)} highly relevant papers!")
            else:
                st.sidebar.warning("No papers found. 可能是条件太苛刻，请检查拼写或放宽条件！")

# --- 导出按钮区 ---
if st.session_state.ai_analyses or st.session_state.cn_summaries:
    st.sidebar.markdown("---")
    st.sidebar.header("💾 导出科研成果")
    csv_bytes = convert_to_csv()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    st.sidebar.download_button(
        label="📥 下载已分析数据 (Excel格式)",
        data=csv_bytes, file_name=f"Pro_Analysis_Data_{current_date}.csv", mime="text/csv"
    )

# --- 主界面展示区 ---
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
                tab1, tab2, tab3, tab4 = st.tabs(["🎯 目的与病原体", "🧪 方法与制备", "📊 结果与局限", "📝 原文摘要"])
                with tab1:
                    st.markdown(f"**Research Purpose:** {res.get('research_purpose')}")
                    st.markdown(f"**Target Pathogens:** {res.get('target_pathogens')}")
                with tab2:
                    st.markdown(f"**Sample Type:** {res.get('sample_type')}")
                    st.markdown(f"**Sample Prep & Extraction:** {res.get('sample_prep_and_extraction')}")
                    st.markdown(f"**Experimental Methods:** {res.get('experimental_methods')}")
                    st.markdown(f"**Primer/Probe Sequences:** {res.get('primer_probe_sequences')}")
                with tab3:
                    st.markdown(f"**Main Results:** {res.get('main_results')}")
                    st.markdown(f"**Limitations:** {res.get('limitations')}")
                with tab4:
                    st.write(res.get('abstract_text'))
else:
    st.info("👈 设置侧边栏条件并点击 'Fetch Summary List' 开始检索。")