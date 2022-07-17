import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from textblob import TextBlob
from google.transliteration import transliterate_text
from deep_translator import GoogleTranslator
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import re
from streamlit_option_menu import option_menu

wc_data = []
translator = GoogleTranslator()
fig = go.Figure()
st.set_page_config(page_title='Sentiment Analysis', page_icon='📊', layout="wide")


@st.cache(allow_output_mutation=True)
def load_data(dataset):
    user_df = pd.read_csv(dataset)
    return user_df


def clean_text(unformatted_text):
    unformatted_text = str(unformatted_text)
    unformatted_text = re.sub(r'@[A-Za-z0-9]+', '', unformatted_text)
    unformatted_text = re.sub(r'#', '', unformatted_text)
    unformatted_text = re.sub(r'&', '', unformatted_text)
    unformatted_text = re.sub(r"'", '', unformatted_text)
    unformatted_text = re.sub(r".", '', unformatted_text)
    unformatted_text = re.sub(r'RT[\s]+', '', unformatted_text)
    unformatted_text = re.sub(r'https?:\/\/\S+', '', unformatted_text)
    return unformatted_text.lower()


def count_plot(x, y):
    st.write("Count Plot")
    layout = go.Layout(
        title='Multiple Reviews Analysis',
        xaxis=dict(title='Category'),
        yaxis=dict(title='Count'), )
    fig.update_layout(dict1=layout, overwrite=True)
    fig.add_trace(go.Bar(name='Multi Reviews', x=x, y=y))
    st.plotly_chart(fig, use_container_width=True)


def pie_plot(count_positive, count_negative, count_neutral):
    st.write("Pie chart")
    df = pd.DataFrame([['Positive', 'Negative', 'Neutral'], [count_positive, count_negative, count_neutral]]).T
    df.columns = ['type', 'count']
    fig2 = px.pie(df, values='count', names='type', title='Overall Sentiment', hole=0.3)
    fig2.update_traces(marker=dict(colors=['green', 'red', 'blue']))
    st.write(fig2)


def word_cloud_plot():
    global wc_data
    st.write("WordCloud")
    st.markdown("<br><br>", True)
    STOPWORDS.update(["i", "i'm", "im", "it", "this", "will", "they", "it's", "for", "a", "the", "them", "to", "'", "."
                      "?", "/", "<", ">", "</", "/>"])
    wc_data = [data for data in wc_data if data not in STOPWORDS]
    word_could_dict = Counter(wc_data)
    wc = WordCloud().generate_from_frequencies(word_could_dict)
    fig3 = plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig3)


def sentiment_over_time_plot(date_present, input_df):
    container = st.container()
    if date_present == "Yes":
        line_df = input_df.groupby([date_opt, 'sentiment']).size().unstack(fill_value=0)
        fig4 = px.line(line_df, x=line_df.index, y=line_df.columns[0:], markers=True,
                       color_discrete_sequence=['red', 'blue', 'green'])
        with container:
            st.write("Sentiment Over Time")
            st.write(fig4)
            st.markdown('---')


def sentiment_by_words(input_df, temp_df):
    global wc_data
    wc_data = [data for data in wc_data if data not in STOPWORDS]
    word_could_dict = Counter(wc_data)
    most_common = word_could_dict.most_common(5)
    for i in range(len(most_common)):
        pos, neg, neu = 0, 0, 0
        for j in range(input_df.shape[0]):
            if most_common[i][0] in temp_df.iloc[j]:
                if input_df.sentiment.iloc[j] == 'Positive':
                    pos += 1
                elif input_df.sentiment.iloc[j] == 'Negative':
                    neg += 1
                elif input_df.sentiment.iloc[j] == 'Neutral':
                    neu += 1
            else:
                continue
        most_common[i] += (pos, neg, neu)
    word_df = pd.DataFrame(most_common, columns=['Word', '', 'Positive', 'Negative', 'Neutral'])
    st.write("Sentiment by words")

    fig5 = px.bar(word_df, y="Word", x=word_df.columns[2:], color_discrete_sequence=['green', 'red', 'blue'],
                  title="Sentiment Analysis By Word", orientation='h')
    st.write(fig5)


st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)



st.write("""
# Sentiment Analysis App
""")

st.write('Sentiment analysis is the interpretation and classification of emotions '
         '(positive, negative and neutral) within text data using text analysis techniques.'
         ' Sentiment analysis tools allow businesses to identify customer sentiment toward products,'
         ' brands or services in online feedback.')

st.markdown("<br>", True)

selected = option_menu(
    menu_title=None,
    options=["Home", "Sentence", "Contact"],
    orientation="horizontal",
    menu_icon="cast",
    icons=["house", "chevron-right", "envelope",]
)

if selected == "Home":
    st.markdown("---")
    st.header('User Input')
    st.write("You can upload a dataset of your reviews for sentiment analysis"
            " or you can try a single sentence sentiment analysis by navigating"
            " to the sentence page!")
    st.markdown("---")
    st.subheader('Mutiple Reviews Analysis')
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    count_positive, count_negative, count_neutral = 0, 0, 0
    st.markdown("---")

    if uploaded_file is not None:
        user_dataframe = load_data(uploaded_file)
        input_df = user_dataframe.copy()
        st.subheader("Your uploaded Dataset")
        st.write(input_df.head())
        st.markdown("---")
        opt = st.selectbox("Select the column you want to perform analysis on!", input_df.columns)
        date_present = st.radio("Do you have a date column in your dataset?", ["No", "Yes"])
        if date_present == "Yes":
            date_opt = st.selectbox("Select the column where date is present!", input_df.columns)
            try:
                input_df[date_opt] = pd.to_datetime(input_df[date_opt]).dt.date
            except:
                st.write("Seems like an invalid date format!")
        temp_df = input_df[opt]
        input_df['sentiment'] = ''
        input_df[opt] = input_df[opt].apply(clean_text)
        for i in range(temp_df.shape[0]):
            wc_data += str(temp_df.iloc[i]).split()
            text = TextBlob(str(temp_df.iloc[i]))
            result = text.sentiment.polarity
            if result >= 0.5:
                count_positive += 1
                text_sentiment = 'Positive'
            elif result < 0:
                count_negative += 1
                text_sentiment = 'Negative'
            else:
                text_sentiment = "Neutral"
                count_neutral += 1
            input_df['sentiment'].iloc[i] = text_sentiment
        total = count_positive + count_negative + count_neutral
        x = ["Positive", "Negative", "Neutral"]
        y = [count_positive, count_negative, count_neutral]
        if st.button("Analyze"):
            st.header("Your Results:")
            st.markdown("---")
            if count_positive == max(count_positive, count_negative, count_neutral):
                st.write("""## Positive
                        Great Work there! Majority of people have a Positive sentiment!""")
            elif count_negative == max(count_positive, count_negative, count_neutral):
                st.write("""## Negative
                        Try improving! Majority of people have a Negative sentiment!""")
            else:
                st.write("""## Neutral
                         Good Work there, but there's room for improvement! Majority of people have Neutral sentiment!""")

            st.markdown('---')
            st.subheader("Your Dashboard")
            st.markdown('---')

            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                # Count plot
                count_plot(x, y)
                # Sentiment by words plot
                sentiment_by_words(input_df, temp_df)
            with fig_col2:
                # Pie Plot
                pie_plot(count_positive, count_negative, count_neutral)
                # Word Cloud
                word_cloud_plot()

            # Sentiment over time plot
            sentiment_over_time_plot(date_present, input_df)

    else:
        st.write("## Upload a Dataset to start visualizing your data for sentiment analysis!")

st.markdown("<br><br>", True)

if selected == "Contact":
    figure_col1, figure_col2, figure_col3 = st.columns(3)

    with figure_col1:
        st.markdown('''
                <div style="box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                                transition: 0.3s;
                                background-color: #141e31">
                      <img src="https://github.com/buildforacause/team/blob/main/homer.png?raw=true" alt="Avatar" style="width:100% ;padding: 16px 16px;">
                      <div style="padding: 16px 16px;">
                        <h4><b>Aniruddha Fale</b></h4>
                        <p>Student</p>
                        <center>
                        <a href="https://www.linkedin.com/in/aniruddha-fale-610897220/">
                        <svg xmlns="http://www.w3.org/2000/svg" style="margin:10px 10px;" width="32" height="32" fill="currentColor" class="bi bi-linkedin" viewBox="0 0 16 16">
      <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
    </svg></a>
                        <a href="https://github.com/aniruddha1607">
    <svg xmlns="http://www.w3.org/2000/svg" style="margin:10px 10px;" width="32" height="32" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
    </svg></a>
                    </center>
                      </div>
                    </div>
    ''', True)

    with figure_col2:
        st.markdown('''
                    <div style="box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                                transition: 0.3s;
                                background-color: #141e31">
                      <img src="https://github.com/buildforacause/team/blob/main/doctor.png?raw=true" alt="Avatar" style="width:100% ;padding: 16px 16px;">
                      <div style="padding: 16px 16px;">
                        <h4><b>Harmit Saini</b></h4>
                        <p>Student</p>
                        <center>
                        <a href="https://www.linkedin.com/in/harmit-saini-09b818211/">
                        <svg xmlns="http://www.w3.org/2000/svg" style="margin:10px 10px;" width="32" height="32" fill="currentColor" class="bi bi-linkedin" viewBox="0 0 16 16">
      <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
    </svg></a>
                        <a href="https://github.com/buildforacause/">
    <svg xmlns="http://www.w3.org/2000/svg" style="margin:10px 10px;" width="32" height="32" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
    </svg></a>
                    </center>
                      </div>
                    </div>
        ''', True)

    with figure_col3:
        st.markdown('''
                    <div style="box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                                transition: 0.3s;
                                background-color: #141e31">
                      <img src="https://github.com/buildforacause/team/blob/main/ralph.png?raw=true" alt="Avatar" style="width:100% ;padding: 16px 16px;">
                      <div style="padding: 16px 16px;">
                        <h4><b>Siddharth Singh</b></h4>
                        <p>Student</p>
                        <center>
                        <a href="https://www.linkedin.com/in/siddharth-singh-046390206/">
                        <svg xmlns="http://www.w3.org/2000/svg" style="margin:10px 10px;" width="32" height="32" fill="currentColor" class="bi bi-linkedin" viewBox="0 0 16 16">
      <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
    </svg></a>
                        <a href="https://github.com/sidsingh0">
    <svg xmlns="http://www.w3.org/2000/svg" style="margin:10px 10px;" width="32" height="32" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
    </svg></a>
                    </center>
                      </div>
                    </div>
        ''', True)


if selected == "Sentence":
    st.subheader('Single Review Analysis')
    st.markdown("Single review analysis helps you to find out the sentiment behind a sentence or a paragraph."
                " The way this section has been written is where you can also check sentiments in <em>HINGLISH</em>!<br>"
                " <br>For example: <strong>Mai ek accha insaan hu</strong> gives a Positive sentiment!"
                " Although it might not work perfectly for some cases!", True)
    single_review = st.text_input('Enter single review below:')
    if single_review:
        text = TextBlob(single_review)
        result = text.sentiment.polarity
        if 0 <= result < 0.5:
            transliterated_text = transliterate_text(single_review, lang_code='hi')
            eng_text = GoogleTranslator(source='auto', target='en').translate(transliterated_text)
            text = TextBlob(eng_text)
            result = text.sentiment.polarity
        if result >= 0.5:
            st.write("""# Positive
                    Great Work there!""")
        elif result < 0:
            st.write("""# Negative
                    Try improving!""")
        else:
            st.write("""# Neutral
                     Good Work there, but there's room for improvement!""")

st.markdown("""<br><center>Created with ❤️ in India by <a href='#' style='text-decoration: None; color: #FF4B4B'>Brown Munde</a>
            </center>""", True)
