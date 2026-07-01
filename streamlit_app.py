import os

import numpy as np
import streamlit as st
from PIL import Image
from plotly.graph_objects import Figure
import plotly.graph_objects as go
from tensorflow.keras.models import load_model


MODEL_PATH = os.path.join("model", "emotion_model.h5")
EMOTION_LABELS = [
    "Angry 😠",
    "Disgust 🤢",
    "Fear 😨",
    "Happy 😊",
    "Neutral 😐",
    "Sad 😢",
    "Surprise 😲",
]

# Tema warna (tetap sesuai emosi)
EMOTION_THEME = {
    "Angry 😠": {"bg": "rgba(239,68,68,.12)", "fg": "#ff5a5f", "accent": "#ef4444"},
    "Disgust 🤢": {"bg": "rgba(139,92,246,.12)", "fg": "#b15cff", "accent": "#ab47bc"},
    "Fear 😨": {"bg": "rgba(59,130,246,.12)", "fg": "#60a5fa", "accent": "#42a5f5"},
    "Happy 😊": {"bg": "rgba(34,197,94,.12)", "fg": "#34d399", "accent": "#22c55e"},
    "Neutral 😐": {"bg": "rgba(148,163,184,.12)", "fg": "#cbd5e1", "accent": "#94a3b8"},
    "Sad 😢": {"bg": "rgba(20,184,166,.12)", "fg": "#2dd4bf", "accent": "#26c6da"},
    "Surprise 😲": {"bg": "rgba(255,193,7,.10)", "fg": "#ffd166", "accent": "#ffb300"},
}


@st.cache_resource(show_spinner=False)
def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan di: {MODEL_PATH}.\n"
            f"Jalankan training dulu: py train_model.py"
        )
    return load_model(MODEL_PATH)


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("L").resize((48, 48))

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (48,48,1)
    img_array = np.expand_dims(img_array, axis=0)  # (1,48,48,1)
    return img_array


def render_dashboard_css():
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
          :root { color-scheme: dark; }
          html, body { background: transparent !important; }

          /* App background */
          .dvDarkWrap {
            background: radial-gradient(1200px 600px at 10% 10%, rgba(124,58,237,.22), transparent 60%),
                        radial-gradient(900px 450px at 90% 20%, rgba(34,211,238,.16), transparent 55%),
                        linear-gradient(180deg, #050814, #070b1f);
            border-radius: 18px;
            padding: 18px;
          }

          /* Glassmorphism */
          .glass {
            background: linear-gradient(180deg, rgba(255,255,255,.085), rgba(255,255,255,.04));
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 18px;
            box-shadow: 0 25px 70px rgba(0,0,0,.40);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
          }

          .cardPad { padding: 18px; }
          .titleH { font-family: 'Inter', sans-serif; font-weight: 900; letter-spacing: -0.6px; }
          .muted { color: rgba(233,236,255,.72); font-size: 13px; }

          /* Buttons */
          .btnNeon {
            border: 1px solid rgba(255,255,255,.14);
            border-radius: 14px;
            background: linear-gradient(90deg, rgba(124,58,237,.95), rgba(34,211,238,.75));
            color: #e9ecff;
            font-weight: 800;
            padding: 12px 16px;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            transition: transform .15s ease, box-shadow .15s ease;
            box-shadow: 0 10px 30px rgba(124,58,237,.16);
          }
          .btnNeon:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 45px rgba(124,58,237,.22);
          }

          .emotionBig { font-size: 34px; font-weight: 900; letter-spacing: -0.4px; }

          /* Progress */
          .progress {
            height: 12px; border-radius: 999px;
            background: rgba(255,255,255,.08);
            border: 1px solid rgba(255,255,255,.10);
            overflow: hidden;
            margin-top: 10px;
          }
          .progress > span {
            display: block; height: 100%; width: 0%;
            background: linear-gradient(90deg, rgba(124,58,237,.95), rgba(34,211,238,.85));
            border-radius: 999px;
            transition: width .6s ease;
          }

          /* Chips grid */
          .grid7 {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
          }
          .emChip {
            padding: 12px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,.12);
            background: rgba(255,255,255,.04);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            transition: transform .15s ease, border-color .15s ease, background .15s ease;
          }
          .emChip:hover {
            transform: translateY(-2px);
            border-color: rgba(124,58,237,.45);
            background: rgba(124,58,237,.07);
          }

          .dot {
            width: 10px; height: 10px; border-radius: 99px;
            background: rgba(124,58,237,.9);
            box-shadow: 0 0 0 4px rgba(124,58,237,.18);
          }

          /* Page cards hover */
          .hoverLift {
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
          }
          .hoverLift:hover {
            transform: translateY(-3px);
            border-color: rgba(34,211,238,.35);
            box-shadow: 0 35px 90px rgba(0,0,0,.48);
          }

          @media (max-width: 900px){
            .grid7 { grid-template-columns: 1fr; }
          }

          /* Premium Hero + Stats */
          .heroGrid{ display:flex; gap:22px; align-items:flex-start; justify-content:space-between; }
          .heroLeft{ flex: 1.1; min-width: 320px; }
          .heroRight{ flex: 0.9; min-width: 340px; }
          @media (max-width: 900px){
            .heroGrid{ flex-direction:column; }
            .heroRight{ width:100%; min-width: unset; }
          }
          .heroPill{ display:inline-flex; align-items:center; gap:10px; padding:10px 14px; border-radius:999px; border:1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.04); color: rgba(233,236,255,.9); font-weight:800; margin-bottom:14px; box-shadow: 0 0 0 1px rgba(124,58,237,.15) inset; }
          .heroTitle{ font-family:'Inter', sans-serif; font-weight:900; font-size:44px; line-height:1.05; letter-spacing:-.8px; margin-top:6px; }
          @media (max-width: 720px){
            .heroTitle{ font-size:30px; }
          }
          .heroDesc{ margin-top:12px; color: rgba(233,236,255,.78); font-size:15px; max-width: 720px; }
          .heroActions{ margin-top:18px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
          .btnHero{ padding:16px 18px; border-radius:16px; font-size:16px; }
          .heroMeta{ color: rgba(233,236,255,.72); font-size:13px; padding:10px 0; }

          .statsGrid{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; }
          @media (max-width: 720px){
            .statsGrid{ grid-template-columns: 1fr; }
          }
          .statCard{ padding:16px; border-radius:18px; border:1px solid rgba(255,255,255,.12); background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.025)); box-shadow: 0 25px 70px rgba(0,0,0,.28); transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
          .statCard:hover{ transform: translateY(-3px); border-color: rgba(34,211,238,.38); box-shadow: 0 40px 95px rgba(0,0,0,.48); }
          .statK{ color: rgba(233,236,255,.70); font-size:12px; font-weight:800; }
          .statV{ font-size:24px; font-weight:900; margin-top:6px; }
          .statS{ font-size:12px; margin-top:6px; }
        </style>

        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Deteksi Emosi Wajah", page_icon="😊", layout="wide")
    render_dashboard_css()

    def get_current_page():
        pages = ["Beranda", "Deteksi", "Riwayat", "Tentang"]
        params = st.query_params
        if "page" in params and params["page"]:
            requested = params["page"][0]
            if requested in pages:
                st.session_state.selected_page = requested
        if "selected_page" not in st.session_state:
            st.session_state.selected_page = "Beranda"
        return st.session_state.selected_page

    page = get_current_page()

    if "next_page" in st.session_state:
        st.session_state.selected_page = st.session_state.next_page
        del st.session_state.next_page

    with st.sidebar:
        st.markdown(
            """
            <style>
              .bbxSidebar {
                padding: 22px 18px;
                background: linear-gradient(180deg, rgba(11,16,40,.75), rgba(7,11,31,.55));
                border-right: 1px solid rgba(255,255,255,.08);
              }
              .bbxBrand{ display:flex; align-items:center; gap:12px; padding:12px 12px; border-radius: 16px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.10); box-shadow: 0 10px 30px rgba(0,0,0,.25); }
              .bbxLogo{ width:42px; height:42px; border-radius: 14px; background: radial-gradient(circle at 30% 30%, rgba(34,211,238,.85), rgba(124,58,237,.9)); border:1px solid rgba(255,255,255,.18); display:grid; place-items:center; font-weight:800; }
              .bbxFooter{ margin-top: 18px; color: rgba(233,236,255,.72); font-size: 12px; text-align:center; }
            </style>
            <div class='bbxSidebar'>
              <div class='bbxBrand'>
                <div class='bbxLogo'>AI</div>
                <div>
                  <div style='font-size:14px; font-weight:800;'>EmoVision</div>
                  <div style='font-size:12px; color: rgba(233,236,255,.72); margin-top:2px;'>Dashboard Emotion CNN</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.sidebar.radio(
            "Menu",
            ["Beranda", "Deteksi", "Riwayat", "Tentang"],
            index=["Beranda", "Deteksi", "Riwayat", "Tentang"].index(page),
            key="selected_page",
        )

        st.markdown("<div class='bbxFooter'>v1.0 • AI Premium UI</div>", unsafe_allow_html=True)

    page = st.session_state.selected_page

    # Konten utama Streamlit; sidebar Streamlit menangani layout secara stabil

    # Pastikan model tersedia sebelum operasi deteksi
    try:
        _ = get_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    def render_home():

        html_code = """
<div class='dvDarkWrap glass cardPad'>
  <div class='heroGrid'>
    <div class='heroLeft'>
      <div class='heroPill'>🧠 AI Dashboard • CNN Inference</div>
      <div class='heroTitle'>Deteksi Emosi Wajah Menggunakan Deep Learning CNN</div>
      <div class='heroDesc'>Upload foto wajah, sistem memprediksi emosi (7 kelas) dan menampilkan confidence score berbasis softmax.</div>
      <div class='heroActions'>
        <div class='heroMeta'>Mode: On-Demand • 48x48 Grayscale</div>
      </div>
    </div>


  </div>
</div>
"""
        st.markdown(html_code, unsafe_allow_html=True)

        # Navigasi Streamlit native (tanpa JS / tanpa postMessage)
        if st.button("🚀 Mulai Deteksi", key="btnMulaiDeteksi", help="Mulai deteksi emosi", use_container_width=False):
            st.session_state.next_page = "Deteksi"
            st.rerun()

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


        st.markdown(
            """
            <div class='glass cardPad'>
              <h2 style='margin:0; font-size:18px;'>Fitur Utama Aplikasi</h2>
              <p class='muted' style='margin:10px 0 0; font-size:13px;'>Tampilan responsif, inferensi cepat, dan pengalaman dashboard AI profesional • Dibuat oleh Junia Sahfitri</p>

              <div style='margin-top:16px; display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 14px;'>
                <div class='emChip'><div><div style='font-weight:900;'>Deep Learning CNN</div><div class='muted' style='margin-top:4px;'>Inference model siap pakai</div></div></div>
                <div class='emChip'><div><div style='font-weight:900;'>Akurasi Tinggi</div><div class='muted' style='margin-top:4px;'>Confidence dari softmax</div></div></div>
                <div class='emChip'><div><div style='font-weight:900;'>Real-time Prediction</div><div class='muted' style='margin-top:4px;'>Respon cepat setelah upload</div></div></div>
                <div class='emChip'><div><div style='font-weight:900;'>Aman dan Privat</div><div class='muted' style='margin-top:4px;'>Diproses lokal di device</div></div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Panel 7 kelas
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='glass cardPad'>
              <h2 style='margin:0; font-size:16px;'>Panel 7 Kelas Emosi</h2>
              <div class='muted' style='margin-top:8px; font-size:13px;'>Angkatan label yang digunakan model.</div>
              <div class='grid7' style='margin-top:14px;'>
            """,
            unsafe_allow_html=True,
        )
        # Grid 7 dengan loop streamlit card (tanpa HTML tambahan biar stabil)
        left, right = st.columns(2, gap="small")
        for idx, label in enumerate(EMOTION_LABELS):
            theme = EMOTION_THEME.get(label)
            target = left if idx % 2 == 0 else right
            with target:
                st.markdown(
                    f"""
                    <div class='emChip' style='margin-bottom:12px; border-radius:18px; background: rgba(255,255,255,.04);'>
                      <div style='display:flex; align-items:center; gap:10px;'>
                        <div class='dot' style='background:{theme['accent']}; box-shadow: 0 0 0 4px {theme['accent']}22;'></div>
                        <div style='font-weight:900; font-size:13px;'>{label}</div>
                      </div>
                      <div class='muted' style='font-weight:800; font-size:12px;'>Kelas</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div></div>", unsafe_allow_html=True)

    def render_detection():
        st.markdown(
            """
            <div class='glass cardPad'>
              <h2 style='margin:0; font-size:18px;'>Upload Foto</h2>
              <p class='muted' style='margin:10px 0 0; font-size:13px;'>Drag & Drop, pilih gambar, lalu prediksi emosi dengan CNN.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Drag & Drop atau Pilih Gambar",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
        )

        if not uploaded_file:
            st.info("Upload foto terlebih dahulu untuk melihat prediksi emosi.")
            return

        try:
            pil_img = Image.open(uploaded_file)
        except Exception:
            st.error("Format gambar tidak didukung.")
            return

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        left_col, right_col = st.columns([1, 1.2], gap="large")
        with left_col:
            st.image(pil_img, caption="Preview", use_container_width=True)
            with st.expander("Tips agar hasil lebih akurat"):
                st.write("- Pastikan wajah terlihat jelas")
                st.write("- Pencahayaan terang membantu model")
                st.write("- Hindari gambar terlalu blur")

        with right_col:
            model = get_model()
            img_array = preprocess_image(pil_img)

            # Loading animation sederhana (tanpa JS kompleks di Streamlit)
            with st.spinner("AI sedang menganalisis..."):
                prediction = model.predict(img_array, verbose=0)

            predicted_class = int(np.argmax(prediction, axis=1)[0])
            emotion = EMOTION_LABELS[predicted_class]
            confidence = float(np.max(prediction)) * 100
            probs = prediction[0].astype(float)

            # Riwayat
            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.insert(0, {
                "emotion": emotion,
                "confidence": confidence,
                "filename": getattr(uploaded_file, 'name', 'upload'),
                "img": pil_img,
            })
            st.session_state.history = st.session_state.history[:5]

            theme = EMOTION_THEME.get(emotion, {"fg": "#e9ecff", "accent": "#7c3aed"})

            st.markdown(
                f"""
                <div class='glass cardPad'>
                  <div class='muted'>Emosi terdeteksi</div>
                  <div class='emotionBig' style='color:{theme['fg']}; margin-top:4px;'>{emotion}</div>
                  <div class='muted' style='margin-top:6px;'>Confidence: <b style='color:{theme['fg']}'>{confidence:.2f}%</b></div>
                  <div class='progress'><span style='width:{confidence:.2f}%;'></span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.caption("Probabilitas seluruh kelas")

            pairs = list(zip(EMOTION_LABELS, probs.tolist()))
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

            chips_html = "<div class='grid7'>"
            for name, prob in pairs_sorted:
                pct = prob * 100
                chips_html += (
                    "<div class='emChip'>"
                    "  <div style='display:flex; align-items:center; gap:10px;'>"
                    "    <div class='dot'></div>"
                    f"    <div style='font-weight:800; font-size:13px;'>{name}</div>"
                    "  </div>"
                    f"  <div class='muted' style='font-weight:800; font-size:12px; color:rgba(233,236,255,.9);'>{pct:.2f}%</div>"
                    "</div>"
                )
            chips_html += "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)

    def render_history():
        if "history" not in st.session_state or not st.session_state.history:
            st.info("Belum ada riwayat prediksi.")
            return

        st.markdown("<div class='glass cardPad'><h2 style='margin:0; font-size:18px;'>Riwayat Prediksi</h2><div class='muted' style='margin-top:8px;'>5 prediksi terakhir</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        for item in st.session_state.history:
            theme = EMOTION_THEME.get(item['emotion'], {"fg": "#e9ecff", "accent": "#7c3aed"})
            with st.container():
                st.markdown(
                    f"""
                    <div class='glass cardPad' style='margin-bottom:14px;'>
                      <div style='display:flex; gap:14px; align-items:flex-start; flex-wrap:wrap;'>
                        <div style='min-width:140px; max-width:220px;'>
                          <div class='muted'>Gambar</div>
                        </div>
                        <div style='flex:1;'>
                          <div class='muted'>Emosi terdeteksi</div>
                          <div class='emotionBig' style='font-size:26px; color:{theme['fg']}; margin-top:4px;'>{item['emotion']}</div>
                          <div class='muted' style='margin-top:6px;'>Confidence: <b style='color:{theme['fg']}'>{item['confidence']:.2f}%</b></div>
                          <div class='progress'><span style='width:{item['confidence']:.2f}%;'></span></div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.image(item['img'], caption=item.get('filename','upload'), width=260)

    def render_about():
        st.markdown("<div class='dvDarkWrap glass cardPad'><div class='titleH' style='font-size:28px;'>Tentang</div><div class='muted' style='margin-top:8px;'>Deteksi emosi wajah berbasis CNN (7 kelas) dengan tampilan dashboard AI premium.</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='glass cardPad'>
              <h2 style='margin:0; font-size:18px;'>Komponen Utama</h2>
              <ul style='margin-top:10px; color: rgba(233,236,255,.72)'>
                <li>Model: <b>model/emotion_model.h5</b></li>
                <li>Preprocessing: 48x48 grayscale</li>
                <li>Inference: Softmax menghasilkan probabilitas per kelas</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render berdasarkan menu
    page = st.session_state.selected_page
    if page == "Beranda":
        render_home()
    elif page == "Deteksi":
        render_detection()
    elif page == "Riwayat":
        render_history()
    else:
        render_about()

    st.markdown("</div>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()


