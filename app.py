from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        k = request.form.get("k")

        # VALIDASI FILE
        if not file or file.filename == "":
            return "❌ File belum dipilih"

        # VALIDASI K
        try:
            k = int(k)
        except:
            return "❌ Jumlah cluster harus angka"

        try:
            # BACA CSV
            df = pd.read_csv(file, encoding="utf-8")

            if df.empty:
                return "❌ File kosong"

            # VALIDASI KOLOM
            required_cols = ["Wilayah", "Mobil", "Bus", "Truk", "Motor"]
            for col in required_cols:
                if col not in df.columns:
                    return f"❌ Kolom '{col}' tidak ada di CSV"

            # AMBIL FITUR
            X = df[["Mobil", "Bus", "Truk", "Motor"]]

            # NORMALISASI
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # K-MEANS
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df["Cluster"] = kmeans.fit_predict(X_scaled)

            # TAMBAH TOTAL
            df["Total"] = df[["Mobil", "Bus", "Truk", "Motor"]].sum(axis=1)

            # INSIGHT
            total_motor = int(df["Motor"].sum())
            wilayah_tertinggi = df.loc[df["Motor"].idxmax()]["Wilayah"]

            # BUAT FOLDER STATIC (kalau belum ada)
            if not os.path.exists("static"):
                os.makedirs("static")

            # VISUALISASI
            plt.figure()
            plt.scatter(df["Motor"], df["Mobil"], c=df["Cluster"])
            plt.xlabel("Motor")
            plt.ylabel("Mobil")
            plt.title("Clustering Kendaraan Jawa Barat")
            plt.savefig("static/plot.png")
            plt.close()

            data = df.to_dict(orient="records")

            return render_template(
                "hasil.html",
                data=data,
                total_motor=total_motor,
                wilayah_tertinggi=wilayah_tertinggi
            )

        except Exception as e:
            return f"❌ ERROR: {str(e)}"

    return render_template("index.html")


# WAJIB UNTUK DEPLOY (Render / Railway)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
