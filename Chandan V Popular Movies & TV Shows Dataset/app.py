from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st


st.set_page_config(
    page_title="StreamScope | Content intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    path = Path(__file__).with_name("streaming_content_trends.csv")
    data = pd.read_csv(path)
    data["release_date"] = pd.to_datetime(data["release_date"], errors="coerce")
    data["release_year"] = data["release_year"].astype("Int64")
    data["genres"] = data["genres"].fillna("Unknown")
    data["original_language"] = data["original_language"].fillna("Unknown")
    data["search_category"] = data["search_category"].str.replace("_", " ").str.title()
    return data


def explode_genres(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(genre=data["genres"].str.split(", ")).explode("genre")


@st.cache_data
def build_genre_summary(data: pd.DataFrame) -> pd.DataFrame:
    genre_data = explode_genres(data)
    return (
        genre_data.groupby("genre", as_index=False)
        .agg(titles=("title", "nunique"), average_rating=("vote_average", "mean"), popularity=("popularity", "mean"))
        .sort_values("titles", ascending=False)
    )


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #18222d; --muted: #667482; --mint: #1fbf9f; --coral: #f26b5e; --paper: #f7f8f4; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    h1 { font-size: 2.65rem !important; line-height: 1.05 !important; margin-bottom: .2rem !important; }
    [data-testid="stSidebar"] { background: #18222d; }
    [data-testid="stSidebar"] * { color: #eef6f1 !important; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e4e9e5; padding: 1rem 1.1rem; border-radius: 8px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    .eyebrow { color: var(--mint); font-size: .75rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .subtitle { color: var(--muted); font-size: 1.05rem; margin-bottom: 1.5rem; }
    .section-label { color: var(--muted); font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin: 1.8rem 0 .7rem; }
    .insight { background: #e6f4ee; border-left: 4px solid var(--mint); padding: .9rem 1rem; border-radius: 0 6px 6px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


data = load_data()
min_year = int(data["release_year"].min())
max_year = int(data["release_year"].max())

with st.sidebar:
    st.markdown("## STREAMSCOPE")
    st.caption("A practical read on the catalog's momentum, taste, and shape.")
    st.markdown("### Refine the catalog")
    media_types = st.multiselect("Format", sorted(data["media_type"].dropna().unique()), default=sorted(data["media_type"].dropna().unique()))
    categories = st.multiselect("Signal", sorted(data["search_category"].dropna().unique()), default=sorted(data["search_category"].dropna().unique()))
    languages = st.multiselect("Language", sorted(data["original_language"].dropna().unique()), default=[])
    year_range = st.slider("Release year", min_year, max_year, (min_year, max_year))
    min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.1)
    recent_only = st.checkbox("Recent releases only", value=False)

filtered = data[
    data["media_type"].isin(media_types)
    & data["search_category"].isin(categories)
    & data["release_year"].between(*year_range)
    & (data["vote_average"] >= min_rating)
]
if languages:
    filtered = filtered[filtered["original_language"].isin(languages)]
if recent_only:
    filtered = filtered[filtered["is_recent"]]

st.markdown('<div class="eyebrow">Catalog intelligence / 2026 edition</div>', unsafe_allow_html=True)
st.title("What is moving the catalog?")
st.markdown('<div class="subtitle">Explore the titles, genres, and signals shaping this streaming snapshot.</div>', unsafe_allow_html=True)

if filtered.empty:
    st.warning("No titles match these filters. Widen the year, rating, or format selection to continue.")
    st.stop()

top_title = filtered.sort_values(["popularity", "vote_count"], ascending=False).iloc[0]
genre_summary = build_genre_summary(filtered)
top_genre = genre_summary.iloc[0]

kpi_cols = st.columns(4)
kpi_cols[0].metric("Titles in view", f"{len(filtered):,}", f"{len(filtered) / len(data):.0%} of catalog")
kpi_cols[1].metric("Avg. audience score", f"{filtered['vote_average'].mean():.1f} / 10", f"{filtered['vote_count'].sum():,} votes")
kpi_cols[2].metric("Avg. popularity", f"{filtered['popularity'].mean():.1f}", f"Peak {filtered['popularity'].max():.1f}")
kpi_cols[3].metric("Leading genre", top_genre["genre"], f"{int(top_genre['titles'])} titles")

st.markdown('<div class="section-label">The shape of demand</div>', unsafe_allow_html=True)
chart_left, chart_right = st.columns([1.15, 1])

with chart_left:
    yearly = filtered.dropna(subset=["release_year"]).groupby("release_year", as_index=False).agg(
        titles=("title", "count"), average_popularity=("popularity", "mean")
    )
    fig_year = alt.Chart(yearly).mark_area(color="#1fbf9f", opacity=0.8).encode(
        x=alt.X("release_year:Q", title=None), y=alt.Y("titles:Q", title="Titles"), tooltip=["release_year", "titles"]
    ).properties(title="Catalog volume by release year", height=280)
    st.altair_chart(fig_year, use_container_width=True)

with chart_right:
    category_counts = filtered["search_category"].value_counts().rename_axis("signal").reset_index(name="titles")
    fig_signal = alt.Chart(category_counts).mark_bar(color="#1fbf9f").encode(
        x=alt.X("titles:Q", title=None), y=alt.Y("signal:N", sort="-x", title=None), tooltip=["signal", "titles"]
    ).properties(title="How titles are surfacing", height=280)
    st.altair_chart(fig_signal, use_container_width=True)

st.markdown('<div class="section-label">Taste map</div>', unsafe_allow_html=True)
taste_left, taste_right = st.columns([1.2, .8])
with taste_left:
    scatter = filtered.copy()
    scatter["short_title"] = scatter["title"].str.slice(0, 28)
    fig_scatter = alt.Chart(scatter).mark_circle(opacity=0.72).encode(
        x=alt.X("vote_average:Q", title="Audience rating"), y=alt.Y("popularity:Q", title="Popularity"),
        size=alt.Size("vote_count:Q", legend=None), color=alt.Color("media_type:N", scale=alt.Scale(domain=["movie", "tv"], range=["#f26b5e", "#1fbf9f"])),
        tooltip=["title", "media_type", "vote_average", "popularity", "vote_count"]
    ).properties(title="Popularity versus audience rating", height=340)
    st.altair_chart(fig_scatter, use_container_width=True)

with taste_right:
    language_counts = filtered["original_language"].value_counts().head(8).rename_axis("language").reset_index(name="titles")
    fig_lang = alt.Chart(language_counts.sort_values("titles")).mark_bar(color="#f26b5e").encode(
        x=alt.X("titles:Q", title=None), y=alt.Y("language:N", sort="-x", title=None), tooltip=["language", "titles"]
    ).properties(title="Top original languages", height=340)
    st.altair_chart(fig_lang, use_container_width=True)

st.markdown('<div class="section-label">Spotlight</div>', unsafe_allow_html=True)
st.markdown(f'<div class="insight"><strong>{top_title["title"]}</strong> leads this view with a popularity score of <strong>{top_title["popularity"]:,.1f}</strong> and an audience rating of <strong>{top_title["vote_average"]:.1f}</strong>. It is a {top_title["media_type"]} in the {top_title["genres"]} lane.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Browse the shortlist</div>', unsafe_allow_html=True)
search = st.text_input("Search titles", placeholder="Try a title, genre, or language", label_visibility="collapsed")
table = filtered.copy()
if search:
    mask = table.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False)).any(axis=1)
    table = table[mask]
table = table.sort_values(["popularity", "vote_average"], ascending=False)
st.dataframe(table[["title", "media_type", "genres", "original_language", "release_year", "vote_average", "vote_count", "popularity"]].head(50), use_container_width=True, hide_index=True, column_config={"vote_average": st.column_config.NumberColumn("Rating", format="%.1f"), "popularity": st.column_config.NumberColumn("Popularity", format="%.1f"), "vote_count": st.column_config.NumberColumn("Votes", format="%,d")})
st.caption(f"Showing {min(len(table), 50):,} of {len(table):,} matching titles. Source: streaming_content_trends.csv")