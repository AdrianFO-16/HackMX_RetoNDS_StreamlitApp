import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_folium import folium_static
import folium

st.set_page_config(
    page_title="Team Cheems",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache
def load_data():
    data = pd.DataFrame(None)
    for i in range(1,16):
        data = pd.concat([data, pd.read_csv("dataset/fraudTrain_"+str(i)+".csv")], axis = 0)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    return data

@st.cache
def visualize_data():
    vc = data['is_fraud'].value_counts()
    fig = px.bar(data, x=vc.index, y=vc.values,
                 labels=dict(x="Transacciones", y="Cantidad"))
    fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [0, 1],
        ticktext = ['Normal', 'Fraude']
        )
    )
    return fig

@st.cache
def con_fraude(data):
    return data[data['is_fraud']==1]


if __name__ == "__main__":    
    
    with st.sidebar:
        st.title("Team Cheems - 31")
        st.header("Navegación")
        mode = st.radio(
            "Menu",
            [
                "Reto NDS",
                "EDA"
            ],
        )
        #st.subheader("Select a state or all US states:")
        #all_states = st.checkbox("All US States", True)
        #locations = np.append(["USA Total"], states)
        #state = st.selectbox("State", states, index=37)
    if mode == 'Reto NDS':
        header = st.beta_container()
        dataset = st.beta_container()
        viz = st.beta_container()
        with header:
            st.title('Detección de fraude')
            st.text('''Este viz app fue creada con la finalidad de explorar los datos
que contiene nuestro dataset y encontrar patrones o anomalías que puedan ser
importantes a tomar encuenta.''')
    
        with dataset:
            st.header('Dataset para detección de fraude')
            data_load_state = st.text('Loading data...')
            data = load_data()
            data_load_state.text("Done!")
            st.subheader('Raw data')
            st.write(data.sample(10))
        
        with viz:
            st.header('Visualización de data no balanceada')
            st.text('''El principal reto acerca de los datos relacionados a transacciones bancarias 
es que se debe de recurrir a medidas como overfitting para que el entrenamiento 
del modelo no se sesgue. Esto se debe a que la proporción de transacciones 
fraudolentas es menor a comparación de las miles que no son fraude.''')
            fig = visualize_data()
            st.plotly_chart(fig)
    
    elif mode == 'EDA':
        data = load_data()
        header = st.beta_container()
        dataset = st.beta_container()
        viz = st.beta_container()
        with header:
            st.title('Análisis Exploratorio')
            st.text('''Se busca conocer los puntos clave que giran en torno a un fraude.''')
            cf = con_fraude(data)
            col1, col2 = st.beta_columns(2)
            col1.header('Variables')
            col1.write(pd.DataFrame(cf.columns))
            col2.header('Fraudes')
            categoria = data.groupby('category')['is_fraud'].sum().sort_values(ascending=False)
            col2.write(categoria)
        with viz:
            st.header('Fraudes por categoría')
            fig = px.bar(categoria, x=categoria.index, y=categoria.values)
            st.plotly_chart(fig)
            es_fraude = data[data['is_fraud'] == 1]
            st.header('Fraudes por sexo')
            sexo = es_fraude['gender'].value_counts()
            fig = px.bar(sexo, x=sexo.index, y=sexo.values)
            st.plotly_chart(fig)
            st.header('Histograma del monto')
            fig = px.histogram(es_fraude, x='amt',color='gender')
            st.plotly_chart(fig)
            st.header('Vendedores con más fraudes')
            vendedor = es_fraude['merchant'].value_counts().nlargest(10)
            fig = px.bar(vendedor, x=vendedor.index, y=vendedor.values)
            st.plotly_chart(fig)
            st.header('Estados con más fraudes')
            estados = es_fraude['state'].value_counts().nlargest(6)
            fig = px.bar(estados, x=estados.index, y=estados.values)
            st.plotly_chart(fig)
        

        
        
