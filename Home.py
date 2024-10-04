import streamlit as st
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="LeanFlow Home",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado ajustado
st.markdown("""
<style>
    body {
        background-color: #1E1E1E;
        color: #E0E0E0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-title {
        font-size: 3rem !important;
        color: #4FC3F7;
        text-align: center;
        margin-top: 0 !important;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-title {
        font-size: 1.5rem;
        color: #B0BEC5;
        text-align: center;
        font-style: italic;
        margin-bottom: 2.5rem;
    }
    .section-title {
        font-size: 1.8rem;
        color: #4FC3F7;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background-color: #4FC3F7;
        color: #1E1E1E;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #03A9F4;
    }
    .footer {
        text-align: center;
        color: #B0BEC5;
        margin-top: 3rem;
        font-size: 0.9rem;
    }
    .footer a {
        color: #4FC3F7;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .feedback-link {
        color: #4FC3F7;
        text-decoration: none;
        font-weight: bold;
    }
    .feedback-link:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar com logo e introdução
image = Image.open('app2.png')
st.sidebar.image(image, width=200) 
st.sidebar.markdown("""
    <h1 style='display: inline; font-size: 28px;'>LeanFlow</h1>
    <h2 style='display: inline; font-size: 18px;'>➤</h2>
    """, unsafe_allow_html=True)
st.sidebar.markdown("""---""")

# Adicionando o link para avaliação
st.sidebar.markdown("""
    <h3>⭐ Avalie nossa aplicação</h3>
    <a href="SEU_LINK_DO_GOOGLE_FORMS_AQUI" target="_blank" class="feedback-link">
        Clique aqui para dar seu feedback
    </a>
    """, unsafe_allow_html=True)

# Cabeçalho principal
st.markdown("<h1 class='main-title'>⚙️ LeanFlow</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Otimizando Processos Hospitalares com Inteligência e Precisão</p>", unsafe_allow_html=True)

# Conteúdo principal
st.markdown("### Sobre a Aplicação", unsafe_allow_html=True)
st.markdown("""
O **LeanFlow** é uma aplicação inovadora que visa revolucionar a análise de processos hospitalares.
Combinando os princípios do **Lean**, **Ciência de Dados** e **Teoria das Restrições (TOC)**,
nós oferecemos ferramentas poderosas para otimizar fluxos de trabalho e melhorar a eficiência operacional.
""")

st.markdown("### Como utilizar este Dashboard?", unsafe_allow_html=True)
st.markdown("""
📊 **Explore os Dados**: Utilize o menu lateral (Ou no botão abaixo) para acessar a aba de **Diagnóstico** e visualizar insights detalhados.<br>
🔍 **Análise Crítica**: Sempre compare os dados gerados com informações de campo para garantir a precisão.<br>
📏 **Medição Precisa**: Garanta que os dados coletados são representativos para resultados mais assertivos.<br>
💡 **Decisões Informadas**: Utilize os insights para promover ações de melhoria contínua nos processos hospitalares.
""", unsafe_allow_html=True)

# Seção interativa
st.markdown("### Comece sua Jornada de Otimização", unsafe_allow_html=True)
if st.button("Iniciar Diagnóstico"):
    st.switch_page("pages/1_Diagnóstico.py")

st.markdown("""
<hr>
<div class='footer' style='text-align: center; padding: 10px; font-size: 0.9em;'>
    © Master Lean Analytics 🦎 | Todos os direitos reservados<br>
    Desenvolvido com ❤️ para melhorar a eficiência hospitalar
</div>
""", unsafe_allow_html=True)