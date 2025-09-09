# FaceRecognition 🔍

A **real-time face recognition and unknown face detection system** built with **PyTorch, OpenCV, and MTCNN**.  
The system authenticates employees against a pre-built database of embeddings and captures unknown faces for future registration.

---

## 🚀 Features
- Employee face **database creation** from images.
- **Real-time face detection** using MTCNN.
- **Face embeddings** generation with FaceNet (InceptionResnetV1).
- **Cosine similarity** matching for recognition.
- Automatic **snapshot saving of unknown faces** for future analysis.

---

## 📂 Project Structure
FaceRecognition/
│── data/
│ ├── employees/ # Employee images (organized by person folder)
│ ├── db/ # Database of face embeddings (.pt file)
│ └── output/ # Unknown snapshots & logs
│── Faceguard.py # Main script (build DB & run detection)
│── requirements.txt # Python dependencies
│── README.md # Documentation

yaml
Copy code

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/FaceRecognition.git
cd FaceRecognition
2. Create a virtual environment (optional but recommended)
bash
Copy code
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/Mac
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
🖼️ Preparing Employee Data
Organize images inside data/employees/ in subfolders by person’s name:

bash
Copy code
data/employees/
├── Alice/
│   ├── alice1.jpg
│   ├── alice2.jpg
├── Bob/
│   ├── bob1.jpg
│   ├── bob2.jpg
▶️ Usage
1. Build Employee Database
Run this to encode employee faces and save embeddings:

bash
Copy code
python Faceguard.py --build-db --db-path data/db/embeddings.pt --employees-dir data/employees
2. Run Real-Time Detection
Run this to start face detection using the webcam (--source 0):

bash
Copy code
python Faceguard.py --detect --db-path data/db/embeddings.pt --source 0
Press q to quit the application.

Unknown faces will be saved automatically in data/output/.

🛠️ Tech Stack
PyTorch → FaceNet embedding model

MTCNN → Face detection & alignment

OpenCV → Real-time video capture & visualization

NumPy / PIL → Image preprocessing
