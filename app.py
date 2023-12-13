import streamlit as st
import pandas as pd
import helper_functions as hf

# Set up page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analyzer", page_icon="📊", layout="wide"
)

# Style adjustments
st.markdown("""<style>div.block-container {padding-top:1rem;}</style>""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv('dataset_visualization.csv')

# Function to handle search and update session state
def search_callback():
    search_term = st.session_state.search_term
    num_tweets = st.session_state.num_tweets
    # Filter the DataFrame based on the search term and number of tweets
    filtered = df[df['processed_content'].str.contains(search_term, case=False, na=False)].head(num_tweets)
    st.session_state.filtered_df = filtered

# Initialize filtered_df in session state
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = df

# Sidebar setup
with st.sidebar:
    st.title("Twitter Sentiment Analyzer")

    st.markdown(
        """
        <div style="text-align: justify;">
            This application conducts sentiment analysis on tweets related to a
            specified search term. Due to Twitter's restrictions on API access 
            for data scraping, generating real-time data is not feasible. 
            Therefore, the analysis is based on a pre-existing dataset. The app 
            is designed to categorize tweets into positive or negative sentiments, 
            making it particularly effective for assessing opinions about brands, 
            products, services, companies, or individuals. Please note that it 
            only supports English language tweets.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form(key="search_form"):
        st.subheader("Search Parameters")
        st.text_input("Search term", key="search_term")
        st.slider("Number of tweets", min_value=0, max_value=len(df), key="num_tweets")
        st.form_submit_button(label="Search", on_click=search_callback)
        st.markdown(
            "Note: it may take a while to load the results, especially with a large number of tweets."
        )

    st.markdown("[Github link](https://github.com/SuperAmy99/Sentiment-Analysis)")
    st.markdown("Created by Lintong Li") 


# Function to create the dashboard
def make_dashboard(filtered_df, bar_color, wc_color):
    # First row: Sentiment Plot, Unigram Plot, Bigram Plot
    col1, col2, col3 = st.columns([28, 34, 38])
    
    with col1:
        sentiment_plot = hf.plot_sentiment(filtered_df)
        sentiment_plot.update_layout(height=350, title_x=0.5)
        st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)
    
    with col2:
        top_unigram = hf.get_top_n_gram(filtered_df, ngram_range=(1, 1), n=10)
        unigram_plot = hf.plot_n_gram(top_unigram, title="Top 10 Occurring Words", color=bar_color)
        unigram_plot.update_layout(height=350)
        st.plotly_chart(unigram_plot, theme=None, use_container_width=True)
    
    with col3:
        top_bigram = hf.get_top_n_gram(filtered_df, ngram_range=(2, 2), n=10)
        bigram_plot = hf.plot_n_gram(top_bigram, title="Top 10 Occurring Bigrams", color=bar_color)
        bigram_plot.update_layout(height=350)
        st.plotly_chart(bigram_plot, theme=None, use_container_width=True)

    # Second row: Data Table and Word Cloud
    col1, col2 = st.columns([60, 40])
    
    with col1:
        def sentiment_color(sentiment):
            if sentiment == "Positive":
                return "background-color: #1F77B4; color: white"
            else:
                return "background-color: #FF7F0E"

        st.dataframe(
            filtered_df[["Sentiment", "Tweet"]].style.applymap(
                sentiment_color, subset=["Sentiment"]
            ),
            height=350,
        )
    
    with col2:
        wordcloud = hf.plot_wordcloud(filtered_df, colormap=wc_color)
        st.pyplot(wordcloud)


# Apply custom tab font styling
adjust_tab_font = """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
    }
    </style>
"""
st.write(adjust_tab_font, unsafe_allow_html=True)



# Main content
if "filtered_df" in st.session_state:
    
    tab1, tab2, tab3 = st.tabs(["All", "Positive 😊", "Negative ☹️"])
    with tab1:
        make_dashboard(st.session_state.filtered_df, bar_color="#54A24B", wc_color="Greens")
    with tab2:
        positive_df = st.session_state.filtered_df.query("Sentiment == 'Positive'")
        make_dashboard(positive_df, bar_color="#1F77B4", wc_color="Blues")
    with tab3:
        negative_df = st.session_state.filtered_df.query("Sentiment == 'Negative'")
        make_dashboard(negative_df, bar_color="#FF7F0E", wc_color="Oranges")

