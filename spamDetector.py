import streamlit as st
import pickle
from win32com.client import Dispatch
import pythoncom


def speak(text):
    pythoncom.CoInitialize()
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(text)


model = pickle.load(open('saved_models/spam.pkl', 'rb'))
cv = pickle.load(open('saved_models/vectorizer.pkl', 'rb'))


def main():
    st.title("Email Spam Classification Application")
    st.write("Built with Streamlit & Python")
    activites = ["Classification", "About"]
    choices = st.selectbox("Select Activities", activites)

    if choices == "Classification":
        st.subheader("Classification")
        msg = st.text_area("Enter a text")

        if st.button("Process"):
            print(msg)
            print(type(msg))
            data = [msg]
            print(data)
            vec = cv.transform(data).toarray()
            result = model.predict(vec)

            if result[0] == 0:
                st.markdown(
                    f"""
                    <div style='
                    text-align: center; border: 3px solid #6D28D9; border-radius: 15px; padding: 25px; background: linear-gradient(145deg, #4C1D95, #5B21B6); box-shadow: 0 12px 40px rgba(109, 40, 217, 0.3);
                    font-family: "Bold Addict", sans-serif; position: relative; overflow: hidden; transform: perspective(1000px);'><div style='
                    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
                    background: radial-gradient(circle at center, rgba(109, 40, 217, 0.1) 0%, transparent 70%);
                    opacity: 0.5; z-index: 1;'></div><div style='position: relative; z-index: 2; transform: rotateX(5deg);'>
                    <h2 style='color: #E9D5FF; margin: 0; font-size: 34px; font-weight: 800;text-shadow: 3px 3px 6px rgba(0,0,0,0.3); letter-spacing: 3px;'>
                    This is Not A Spam Email
                    </h2></div></div>
                    """,
                    unsafe_allow_html=True
                )
                speak("This is Not A Spam Email")
            else:
                st.markdown(
                    f"""
                    <div style='
                    text-align: center; border: 3px solid #6D28D9; border-radius: 15px; padding: 25px; background: linear-gradient(145deg, #4C1D95, #5B21B6); box-shadow: 0 12px 40px rgba(109, 40, 217, 0.3);
                    font-family: "Bold Addict", sans-serif; position: relative; overflow: hidden; transform: perspective(1000px);'><div style='
                    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
                    background: radial-gradient(circle at center, rgba(109, 40, 217, 0.1) 0%, transparent 70%);
                    opacity: 0.5; z-index: 1;'></div><div style='position: relative; z-index: 2; transform: rotateX(5deg);'>
                    <h2 style='color: #E9D5FF; margin: 0; font-size: 34px; font-weight: 800;text-shadow: 3px 3px 6px rgba(0,0,0,0.3); letter-spacing: 3px;'>
                    This is A Spam Email
                    </h2></div></div>
                    """,
                    unsafe_allow_html=True
                )
                speak("This is A Spam Email")


main()
