import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv('data_modelling_v2.csv')


def main():

    # Menambahkan kelas CSS pada container sidebar
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            transition: margin-left 200ms;
        }
        .sidebar:hover .sidebar-content {
            margin-left: 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
    # Membuat container sidebar
    sidebar_container = st.empty()

    # Menampilkan/menyembunyikan sidebar berdasarkan hover
    with sidebar_container:
        option = st.sidebar.selectbox('Menu Layanan', ['Home Page','Internet', 'Telepon','TV'])
        
        
    container = st.container()
    if option != "Internet" and option != "Klasifikasi" and option != "Telepon" and option != "TV":
        container.title('Dashboard Analisis Sentimen Layanan Indihome')
        container.write('<div style="text-align: justify;">Selamat datang di dashboard sederhana ini, dalam analisis sentimen Indihome di Twitter, dengan menggunakan metode K-Nearest Neighbors (KNN) untuk menganalisis sentimen dari tweet-tweet yang terkait dengan layanan Indihome. Metode KNN adalah salah satu metode dalam machine learning yang digunakan untuk klasifikasi data berdasarkan kemiripan dengan data pelatihan yang ada.</div><br>', unsafe_allow_html=True)
        container.write('<div style="text-align: justify;">Sumber data berasal Twitter yang terdiri dari 1000 data terkait dengan layanan Indihome. Data tersebut mencakup beragam tweet yang berhubungan dengan pengalaman pengguna terkait Indihome </div><br>', unsafe_allow_html=True)
        container.write('<div style="text-align: justify;">Labelling awal menggunakan metode <i>Lexicon Based</i>. Lexicon yang di gunakan berasal dari Kamus Inset, yang merupakan sumber referensi yang berasal dari tweet. Dengan menggunakan metode KNN didapat akurasi sebesar 76% dengan nilai k=3</div>', unsafe_allow_html=True)
         # Bagian utama di sebelah kanan dengan tata letak rata kiri dan kanan
   
        
    if option == "Internet":
        st.subheader("Data Layanan Internet")
        tab1, tab2 = st.tabs(["Data", "Graph"])
        
        with tab1:
            data_positive_internet = df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_nostem'].str.contains('internet'))]
            data_negative_internet = df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_nostem'].str.contains('internet'))]
            
            data_positive_view =  data_positive_internet[(data_positive_internet['sentiment_predict'] == 1) & (data_positive_internet['tweet_text_prepocessing_nostem'].str.contains('cepat'))]
            data_negative_view =  data_negative_internet[(data_negative_internet['sentiment_predict'] == 0) & (data_negative_internet['tweet_text_prepocessing_nostem'].str.contains('lambat'))]
            
            data_positive_renamed =  data_positive_view.rename(columns={'tweet_text_prepocessing_nostem': 'Tweet', 'sentiment_predict': 'Label'})
            data_negative_renamed = data_negative_view.rename(columns={'tweet_text_prepocessing_stem': 'Tweet', 'sentiment_predict': 'Label'})          
            
            
            jumlah_positive_internet = len(data_positive_internet[(data_positive_internet['sentiment_predict'] == 1) & (data_positive_internet['tweet_text_prepocessing_stem'].str.contains('internet'))])
            jumlah_negatif_internet = len(data_negative_internet[(data_negative_internet['sentiment_predict'] == 0) & (data_negative_internet['tweet_text_prepocessing_stem'].str.contains('internet'))])
                
        
            if not data_positive_internet.empty:
                st.subheader("Data Sentimen positive layanan indihome 'internet'")
                st.write(data_positive_renamed[['Tweet', 'Label']].iloc[0:10])
                st.write("Jumlah sentimen positive pada layanan Internet: ", jumlah_positive_internet, "Data")
                    
                st.subheader("Data Sentimen negative layanan indihome 'internet'")
                st.write(data_negative_renamed[['Tweet', 'Label']].iloc[9:19])
                st.write("Jumlah sentimen negatif pada layanan Internet: ", jumlah_negatif_internet, "Data")
        
        with tab2:
            st.write("Kata kata positif pada 'Internet'")
            # Kata-kata yang ingin ditampilkan dalam grafik bar beserta jumlahnya
            kata_jumlah = {'cepat': 22, 'luas': 33, 'baik': 16, 'kualitas': 11, 'stabil': 9, 'langgan': 5, 'koneksi': 8, 'paket': 13, 'akses': 9}

            # Mengambil frekuensi kata-kata tertentu
            frekuensi_kata = [kata_jumlah[kata] if kata in kata_jumlah else 0 for kata in kata_jumlah.keys()]

            # Membuat bar graph
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.bar(kata_jumlah.keys(),frekuensi_kata)

            # Memberikan judul dan label pada sumbu
            plt.title('Frekuensi Kata-kata positif dengan Kata "Internet"')
            plt.xlabel('Kata-kata')
            plt.ylabel('Frekuensi')

            # Menampilkan bar graph
            st.pyplot(fig)
            
            
            st.write("Kata kata negatif pada 'Internet'")
            # Kata-kata yang ingin ditampilkan dalam grafik bar beserta jumlahnya
            kata_jumlah = { 'lambat': 28,
                            'ganggu': 22,
                            'mati': 32,
                            'koneksi': 17,
                            'masalah': 29,
                            'hilang': 25,
                            'buruk': 23,
                            'rugi': 26,
                            'tagih': 15,
                            'ganti': 14}

            # Mengambil frekuensi kata-kata tertentu
            frekuensi_kata = [kata_jumlah[kata] if kata in kata_jumlah else 0 for kata in kata_jumlah.keys()]

            # Membuat bar graph
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.bar(kata_jumlah.keys(),frekuensi_kata)

            # Memberikan judul dan label pada sumbu
            plt.title('Frekuensi Kata-kata negatif dengan Kata "Internet"')
            plt.xlabel('Kata-kata')
            plt.ylabel('Frekuensi')

            # Menampilkan bar graph
            st.pyplot(fig)

                
    elif option == "Telepon":
        st.subheader("Data Layanan Telepon")
        tab1, tab2 = st.tabs(["Data", "Graph"])
        
        with tab1:
            data_positive_telepon = df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_nostem'].str.contains('telepon'))]
            data_negative_telepon = df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_nostem'].str.contains('telepon'))]
            
            
            data_positive_renamed_telepon =  data_positive_telepon.rename(columns={'tweet_text_prepocessing_nostem': 'Tweet', 'sentiment_predict': 'Label'})
            data_negative_renamed_telepon =  data_negative_telepon.rename(columns={'tweet_text_prepocessing_nostem': 'Tweet', 'sentiment_predict': 'Label'})          
            
            
            jumlah_positive_telepon = len(data_positive_telepon[(data_positive_telepon['sentiment_predict'] == 1) & (data_positive_telepon['tweet_text_prepocessing_stem'].str.contains('internet'))])
            jumlah_negatif_telepon = len(data_negative_telepon[(data_negative_telepon['sentiment_predict'] == 0) & (data_negative_telepon['tweet_text_prepocessing_stem'].str.contains('internet'))])
                
        
            if not data_positive_telepon.empty:
                st.subheader("Data Sentimen positive layanan indihome 'Telepon'")
                st.write(data_positive_renamed_telepon[['Tweet', 'Label']].iloc[12:18])
                st.write("Jumlah sentimen positive pada layanan Telepon: ", jumlah_positive_telepon, "Data")
                    
                st.subheader("Data Sentimen negative layanan indihome 'Telepon'")
                st.write(data_negative_renamed_telepon[['Tweet', 'Label']].iloc[9:19])
                st.write("Jumlah sentimen negatif pada layanan Telepon: ", jumlah_negatif_telepon, "Data")
        
        with tab2:
            st.write("Kata kata positif pada 'Telepon'")
           
            kata_jumlah = {'cepat': 7, 'mudah': 10, 'akses': 5, 'langgan': 8, 'guna': 4,'bantu': 3}

            # Mengambil frekuensi kata-kata tertentu
            frekuensi_kata = [kata_jumlah[kata] if kata in kata_jumlah else 0 for kata in kata_jumlah.keys()]

            # Membuat bar graph
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.bar(kata_jumlah.keys(),frekuensi_kata)

            # Memberikan judul dan label pada sumbu
            plt.title('Frekuensi Kata-kata positif dengan Kata "Telepon"')
            plt.xlabel('Kata-kata')
            plt.ylabel('Frekuensi')

            # Menampilkan bar graph
            st.pyplot(fig)
            
            
            st.write("Kata kata negatif pada 'Telepon'")
            kata_jumlah = {'mati': 27, 'bayar': 19, 'kecewa': 16, 'pindah': 11, 'ganggu': 19,'kabel': 15,'bayar': 14}

            # Mengambil frekuensi kata-kata tertentu
            frekuensi_kata = [kata_jumlah[kata] if kata in kata_jumlah else 0 for kata in kata_jumlah.keys()]
            
            
            # Membuat bar graph
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.bar(kata_jumlah.keys(),frekuensi_kata)

            # Memberikan judul dan label pada sumbu
            plt.title('Frekuensi Kata-kata negatif dengan Kata "Telepon"')
            plt.xlabel('Kata-kata')
            plt.ylabel('Frekuensi')

            # Menampilkan bar graph
            st.pyplot(fig)
            
    # TV                   
    elif option == "TV":
        st.subheader("Data Layanan TV")
        tab1, tab2 = st.tabs(["Data", "Graph"])
        
        with tab1:
            data_positive_tv= df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_nostem'].str.contains('tv'))]
            data_negative_tv = df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_nostem'].str.contains('tv'))]
                
                
            data_positive_renamed_tv =  data_positive_tv.rename(columns={'tweet_text_prepocessing_nostem': 'Tweet', 'sentiment_predict': 'Label'})
            data_negative_renamed_tv =  data_negative_tv.rename(columns={'tweet_text_prepocessing_nostem': 'Tweet', 'sentiment_predict': 'Label'})          
                
                
            jumlah_positive_tv = len(data_positive_tv[(data_positive_tv['sentiment_predict'] == 1) & (data_positive_tv['tweet_text_prepocessing_nostem'].str.contains('tv'))])
            jumlah_negatif_tv = len(data_negative_tv[(data_negative_tv['sentiment_predict'] == 0) & (data_negative_tv['tweet_text_prepocessing_nostem'].str.contains('tv'))])
                    
            
            if not data_positive_tv.empty:
                st.subheader("Data Sentimen positive layanan indihome 'TV'")
                st.write(data_positive_renamed_tv[['Tweet', 'Label']].iloc[135:140])
                st.write("Jumlah sentimen positive pada layanan TV: ", jumlah_positive_tv , "Data")
                        
                st.subheader("Data Sentimen negative layanan indihome 'TV'")
                st.write(data_negative_renamed_tv[['Tweet', 'Label']].iloc[100:110])
                st.write("Jumlah sentimen negatif pada layanan TV: ", jumlah_negatif_tv , "Data")
    
        with tab2:
            st.write("Kata kata positif pada 'TV'")
            kata_jumlah = {'cepat': 43, 'lancar': 32, 'mudah': 37, 'akses': 29, 'langgan': 22,'tampil': 19, 'konten': 27, 'tayang': 26}

            # Mengambil frekuensi kata-kata tertentu
            frekuensi_kata = [kata_jumlah[kata] if kata in kata_jumlah else 0 for kata in kata_jumlah.keys()]
            
            
            # Membuat bar graph
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.bar(kata_jumlah.keys(),frekuensi_kata)

            # Memberikan judul dan label pada sumbu
            plt.title('Frekuensi Kata-kata positif dengan Kata "TV"')
            plt.xlabel('Kata-kata')
            plt.ylabel('Frekuensi')
            st.pyplot(fig)     
            
            
            st.write("Kata kata negatif pada 'TV'")
            kata_jumlah = {'mati': 42, 'ganggu': 35, 'lambat': 22, 'koneksi': 11, 'sinyal': 17,'rugi': 25,'masalah': 33, 'suara': 17, 'modem': 7, 'gambar': 9}

            # Mengambil frekuensi kata-kata tertentu
            frekuensi_kata = [kata_jumlah[kata] if kata in kata_jumlah else 0 for kata in kata_jumlah.keys()]
            
            
            # Membuat bar graph
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.bar(kata_jumlah.keys(),frekuensi_kata)

            # Memberikan judul dan label pada sumbu
            plt.title('Frekuensi Kata-kata negatif dengan Kata "TV"')
            plt.xlabel('Kata-kata')
            plt.ylabel('Frekuensi')

            # Menampilkan bar graph
            st.pyplot(fig)             
if __name__ == '__main__':
    main()