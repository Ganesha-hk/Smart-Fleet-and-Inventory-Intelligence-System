# Smart Fleet and Inventory Intelligence System 🚛📦

A state-of-the-art solution for real-time fleet tracking, predictive inventory management, and data-driven logistics optimization. This system leverages advanced machine learning to provide actionable insights for fleet operators and warehouse managers.

## ✨ Key Features

- **Real-time Fleet Monitoring**: Track vehicle locations, status, and performance metrics in real-time using interactive maps.
- **Predictive Inventory Intelligence**: AI-powered demand forecasting and stock level optimization to reduce overhead and prevent stockouts.
- **Interactive Dashboards**: High-performance visualizations using Recharts and Framer Motion for a fluid, responsive user experience.
- **ML-Driven Analytics**: Advanced data processing pipelines (PCA, Regression) to identify trends and anomalies in supply chain operations.
- **Unified API**: Scalable FastAPI backend providing robust endpoints for data ingestion and retrieval.

## 🛠️ Technology Stack

### Frontend
- **Framework**: [React 19](https://react.dev/)
- **Build Tool**: [Vite 8](https://vitejs.dev/)
- **Styling**: [Tailwind CSS 4](https://tailwindcss.com/)
- **Animations**: [Framer Motion](https://www.framer.com/motion/)
- **Maps**: [React Leaflet](https://react-leaflet.js.org/)
- **Charts**: [Recharts](https://recharts.org/)
- **State Management**: [Zustand](https://github.com/pmndrs/zustand)

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Language**: Python 3.x
- **Validation**: Pydantic v2
- **Data Science**: Pandas, NumPy, Scikit-learn
- **API Documentation**: Swagger UI (built-in)

## 📁 Project Structure

```text
.
├── backend/            # FastAPI application
│   ├── app/            # Core logic, models, and API routes
│   └── tests/          # Backend unit tests
├── frontend/           # React + Vite application
│   ├── src/            # Components, pages, and hooks
│   └── public/         # Static assets
├── ml_engine/          # Machine Learning pipelines and artifacts
├── data/               # Data storage (raw and processed)
├── infra/              # Infrastructure and deployment scripts
└── scripts/            # Utility scripts for development
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 20+
- npm or yarn

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   pip install -r requirements.txt
   ```
3. Run the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## 📈 ML Pipeline
The system includes a sophisticated ML pipeline located in `ml_engine/`. It handles:
- Feature engineering and data preprocessing.
- Dimensionality reduction using PCA.
- Model training and serialized artifact management.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Developed with ❤️ by [Ganesha-hk](https://github.com/Ganesha-hk)
