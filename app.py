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

        # VALIDASI INPUT
        if not file or file.filename == "":
            return "❌ File belum dipilih"

        try:
            k = int(k)
        except:
            return "❌ Jumlah cluster harus angka"

        try:
            # BACA CSV
            df = pd.read_csv(file, encoding='utf-8')

            # VALIDASI KOLOM
            required_cols = ["Mobil", "Bus", "Truk", "Motor"]
            for col in required_cols:
                if col not in df.columns:
                    return f"❌ Kolom '{col}' tidak ditemukan di file CSV"

            # AMBIL FITUR
            X = df[required_cols]

            # NORMALISASI
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # MODEL KMEANS
            kmeans = KMeans(n_clusters=k, random_state=42)
            df["Cluster"] = kmeans.fit_predict(X_scaled)

            # TAMBAH TOTAL
            df["Total"] = df[required_cols].sum(axis=1)

            # INSIGHT
            total_motor = int(df["Motor"].sum())
            wilayah_tertinggi = df.loc[df["Motor"].idxmax()]["Wilayah"]

            # VISUALISASI
            if not os.path.exists("static"):
                os.makedirs("static")

            plt.figure(figsize=(8,5))
            plt.scatter(df["Motor"], df["Mobil"], c=df["Cluster"], cmap='viridis', s=100)
            plt.xlabel("Motor")
            plt.ylabel("Mobil")
            plt.title("Clustering Kendaraan Jawa Barat")
            plt.grid(True)
            plt.savefig("static/plot.png")
            plt.close()

            data = df.to_dict(orient="records")

            return render_template("hasil.html",
                                   data=data,
                                   total_motor=total_motor,
                                   wilayah_tertinggi=wilayah_tertinggi)

        except Exception as e:
            return f"❌ ERROR: {str(e)}"

    return render_template("index.html")


# WAJIB UNTUK DEPLOY
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)