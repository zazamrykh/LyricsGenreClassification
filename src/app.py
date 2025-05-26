import streamlit as st

# Пока заглушка для предсказания жанра
def predict_genre(lyrics: str) -> str:
    """
    Stub function for predicting song genre based on lyrics.
    Replace this with your actual model inference code.
    """
    # TODO: добавить сюда либо вызов модели напрямую, либо стучаться к бэку
    return "Unknown Genre"

# Заголовок страницы
st.set_page_config(page_title="Song Genre Predictor", page_icon="🎵")

def main():
    st.title("🎶 Song Genre Predictor")
    st.write("Enter song lyrics below and click **Predict** to see the predicted genre.")

    # Поле для ввода текста песни
    lyrics = st.text_area("Song Lyrics", height=200)

    if st.button("Predict"):
        if lyrics.strip() == "":
            st.error("Please enter some lyrics to predict the genre.")
        else:
            genre = predict_genre(lyrics)
            st.subheader("Predicted Genre:")
            st.success(genre)

if __name__ == "__main__":
    main()
