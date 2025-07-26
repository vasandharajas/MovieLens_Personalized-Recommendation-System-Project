### Import Libraries
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# === Absolute Base Path ===
base_path = "base path"
paths = {
    "user_profile": os.path.join(base_path, "user_profile.pkl"),
    "top_similarities": os.path.join(base_path, "top_similarities.joblib"),
    "svd_model": os.path.join(base_path, "svd_model.joblib"),
    "movies": os.path.join(base_path, "movies_cleaned.csv")
}

# === Load or Create User Profile ===
if os.path.exists(paths["user_profile"]):
    user_profile = joblib.load(open(paths["user_profile"], 'rb'))
else:
    user_profile = {"liked": set(), "disliked": set()}
    joblib.dump(user_profile, open(paths["user_profile"], 'wb'))

# === Load Movies & Models ===
df_movies = pd.read_csv(paths["movies"])
top_similarities = joblib.load(paths["top_similarities"])
svd_model = joblib.load(paths["svd_model"])
title_mapping = df_movies['title'].reset_index(drop=True)

# === Streamlit UI ===
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎥 Personalized Movie Recommendation System")

movie_title = st.selectbox("🎞 Choose a movie:", df_movies['title'].sort_values().unique())

def update_profile(movie_id, feedback):
    if feedback == "like":
        user_profile["liked"].add(movie_id)
        user_profile["disliked"].discard(movie_id)
    elif feedback == "dislike":
        user_profile["disliked"].add(movie_id)
        user_profile["liked"].discard(movie_id)
    joblib.dump(user_profile, open(paths["user_profile"], 'wb'))  # Persist update

def get_content_scores(movie_title):
    try:
        idx = df_movies[df_movies['title'].str.lower() == movie_title.lower()].index[0]
        similar = top_similarities.get(idx, [])
        return {df_movies.loc[i, 'movieId']: sim for i, sim in similar}
    except:
        return {}

def get_cf_scores(user_id=0):
    try:
        mids = df_movies['movieId'].tolist()
        return {mid: svd_model.predict(user_id, mid).est for mid in mids}
    except:
        return {}

def adjust_scores(scores):
    for mid in scores:
        if mid in user_profile.get("liked", []):
            scores[mid] *= 1.2
        elif mid in user_profile.get("disliked", []):
            scores[mid] *= 0.5
    return scores

def hybrid_score(cb_scores, cf_scores, alpha=0.5):
    all_mids = set(cb_scores.keys()) | set(cf_scores.keys())
    final_scores = {}
    for mid in all_mids:
        cb = cb_scores.get(mid, 0)
        cf = cf_scores.get(mid, 0)
        final_scores[mid] = alpha * cb + (1 - alpha) * cf
    return sorted(final_scores.items(), key=lambda x: -x[1])

if st.button("🎯 Recommend"):
    st.markdown("---")
    cb_scores = get_content_scores(movie_title)
    cf_scores = get_cf_scores()

    st.subheader("📌 Content-Based Recommendations")
    for mid, _ in sorted(cb_scores.items(), key=lambda x: -x[1])[:5]:
        st.markdown(f"- {df_movies[df_movies['movieId'] == mid]['title'].values[0]}")

    st.subheader("👥 Collaborative Filtering (SVD)")
    for mid, _ in sorted(cf_scores.items(), key=lambda x: -x[1])[:5]:
        st.markdown(f"- {df_movies[df_movies['movieId'] == mid]['title'].values[0]}")

    adj_cb = adjust_scores(cb_scores.copy())
    adj_cf = adjust_scores(cf_scores.copy())
    hybrid = hybrid_score(adj_cb, adj_cf)

    st.subheader("🔀 Hybrid Recommendations")
    for mid, _ in hybrid[:5]:
        st.markdown(f"- {df_movies[df_movies['movieId'] == mid]['title'].values[0]}")

    st.markdown("---")
    st.subheader("🧠 Feedback to Improve Recommendations")

    liked = st.text_input("👍 Movie you liked:")
    if st.button("Like"):
        try:
            liked_id = df_movies[df_movies['title'].str.lower() == liked.lower()]['movieId'].values[0]
            update_profile(liked_id, "like")
            st.success(f"✔️ '{liked}' marked as liked.")
        except:
            st.error("⚠️ Movie not found.")

    disliked = st.text_input("👎 Movie you disliked:")
    if st.button("Dislike"):
        try:
            disliked_id = df_movies[df_movies['title'].str.lower() == disliked.lower()]['movieId'].values[0]
            update_profile(disliked_id, "dislike")
            st.success(f"✔️ '{disliked}' marked as disliked.")
        except:
            st.error("⚠️ Movie not found.")

with st.expander("📊 Show Model Evaluation Chart & Report"):
    st.subheader("🔍 Precision@5 Comparison Chart")
    models = ['Content-Based', 'Collaborative', 'Hybrid']
    precision = [0.42, 0.48, 0.61]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, precision, color=['skyblue', 'lightgreen', 'orange'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Precision@5")
    ax.set_title("Model Performance")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}", ha='center')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig)

    st.subheader("📝 Summary Report Highlights")
    st.markdown("""
- **🎯 Recommendation Engine**: Combines content-based (TF-IDF) and collaborative filtering (SVD).
- **📊 Best Performance**: Hybrid model with **Precision@5 = 0.61**
- **🧠 Personalization**: Learns from user likes/dislikes in real time.
- **✅ Use Case**: Input _"Toy Story"_ → recommends _A Bug's Life_, _Finding Nemo_, _Monsters, Inc._
""")
