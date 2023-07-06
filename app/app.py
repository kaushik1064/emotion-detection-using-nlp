import streamlit as st
import altair as alt

import pandas as pd
import numpy as np

import joblib

import webbrowser 

pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_03_july_2023.pkl","rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def predict_emotions_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Emotion Music recommender")
    menu=["Home", "Monitor","About"]
    choice=st.sidebar.selectbox("Menu",menu)
    lang = st.text_input('Language')
    singer = st.text_input('Singer')
    btn = st.button('recommend me a song')

    if choice=="Home":
        st.subheader("Home-Emotion In Text")

        with st.form(key ='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = predict_emotions_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.write(prediction)
                st.write("Confidence:{}".format(np.max(probability)))

                st.success("Prediction")

            with col2:
                st.success("Predicted Probability")
                st.write(probability)   
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x="emotions",y="probability")
                st.altair_chart(fig,use_container_width=True)
                st.write(proba_df.T)

    if btn:
        webbrowser.open(f"https://www.youtube.com/results?search_query={singer}+{lang}+{predict_emotions(raw_text)}+songs")

    elif choice=="Monitor":
        st.subheader("Monitor App")
    
    else:
        st.subheader("About")


if __name__ == '__main__':
    main()