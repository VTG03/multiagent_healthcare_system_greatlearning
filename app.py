from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Multi-Agent Healthcare Assistant", page_icon="🩺", layout="centered")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 880px;}
.card {
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 16px;
  padding: 1rem 1.2rem;
  background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
  box-shadow: 0 8px 18px rgba(0,0,0,.05);
}
.hint { color: #6b7280; font-size: .92rem; }
.section-title { font-weight: 600; margin: .25rem 0 .5rem; }
.kv { font-size: .95rem; }
hr { border: none; border-top: 1px solid rgba(0,0,0,.06); margin: .8rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("🩺 Healthcare Assistant")
st.caption("Upload an X-ray to get a quick triage and next steps.")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Upload")
c1, c2 = st.columns(2)
with c1:
    img = st.file_uploader("Chest X-ray (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=False, key="img_upl")
    if not img:
        st.info("Select an image to continue.")
with c2:
    pdf = st.file_uploader("PDF report ", type=["pdf"], accept_multiple_files=False, key="pdf_upl")

pc1, pc2 = st.columns(2)
with pc1:
    st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
    if img:
        st.image(img, use_column_width=True)
    else:
        st.empty()
with pc2:
    st.markdown('<div class="section-title">Attachments</div>', unsafe_allow_html=True)
    if pdf:
        st.success(f"PDF: {pdf.name}")
    else:
        st.caption("No PDF attached.")
st.markdown('</div>', unsafe_allow_html=True)

go = st.button("Proceed ➜", type="primary")
if go:
    if not img:
        st.error("Please upload an image.")
        st.stop()

    try:
        from graph.pipeline import build_graph
    except Exception as e:
        st.error(f"Import error: {e}")
        st.stop()

    with st.spinner("Analyzing…"):
        try:
            app = build_graph()
            result = app.invoke({"xray_file": img, "pdf_file": pdf})
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Ingestion")
    st.json(result.get("ingestion", {}))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Imaging")
    st.json(result.get("imaging", {}))
    st.markdown('</div>', unsafe_allow_html=True)

    if "doctor" in result:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Doctor")
        doc = result["doctor"]
        st.warning(doc.get("message", "Escalated to a doctor."))
        st.json(doc)
        st.markdown('</div>', unsafe_allow_html=True)

    elif "therapy" in result:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Therapy")
        th = result["therapy"]
        st.json(th)
        st.markdown('</div>', unsafe_allow_html=True)

        if result.get("pharmacy"):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Nearby Pharmacies")
            ph = result["pharmacy"]

            st.markdown('<div class="kv">', unsafe_allow_html=True)
            st.write(f"**Requested SKUs:** {', '.join(ph.get('requested_skus', [])) or '—'}")
            st.write(f"**Patient location:** {ph.get('patient_location', {}) or '—'}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("**Best nearby**")
            st.json(ph.get("best_nearby", {}))

            st.markdown("**Best price**")
            st.json(ph.get("best_price", {}))

            st.markdown("**Top matches**")
            st.json(ph.get("all_matches", []))

            if ph.get("reservation"):
                st.success("Items held for pickup")
                st.json(ph["reservation"])

            with st.expander("Details"):
                st.write(ph.get("note"))
                st.json(ph.get("debug", {}))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No pharmacy suggestions available.")

    else:
        st.info("No route chosen. Check imaging above.")

st.markdown("<div class='hint'>Flow: Upload → Imaging → Doctor or Therapy → Pharmacy.</div>", unsafe_allow_html=True)




