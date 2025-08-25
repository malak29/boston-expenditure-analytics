import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, classification_report, accuracy_score
import joblib
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import asyncio
import json

from app.core.config import settings
from app.core.exceptions import MLModelException, DataProcessingException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.database import ExpenditureRecord, MLModelMetadata

class MLService:
    
    def __init__(self):
        self.models_path = settings.ml_models_path
        os.makedirs(self.models_path, exist_ok=True)
        self.encoders = {}
        self.scalers = {}
    
    async def prepare_clustering_data(
        self,
        session: AsyncSession,
        fiscal_year: Optional[int] = None
    ) -> pd.DataFrame:
        try:
            query = select(
                ExpenditureRecord.vendor_name,
                ExpenditureRecord.account_descr,
                ExpenditureRecord.dept_name,
                ExpenditureRecord.fiscal_month,
                ExpenditureRecord.fiscal_year,
                ExpenditureRecord.monetary_amount
            )
            
            if fiscal_year:
                query = query.where(ExpenditureRecord.fiscal_year == fiscal_year)
            
            result = await session.execute(query)
            data = result.all()
            
            df = pd.DataFrame([
                {
                    "vendor_name": row.vendor_name,
                    "account_descr": row.account_descr,
                    "dept_name": row.dept_name,
                    "fiscal_month": row.fiscal_month,
                    "fiscal_year": row.fiscal_year,
                    "monetary_amount": row.monetary_amount
                }
                for row in data
            ])
            
            if df.empty:
                raise DataProcessingException("No data available for clustering")
            
            return df
            
        except Exception as e:
            raise DataProcessingException("clustering data preparation", {"error": str(e)})
    
    def perform_kmeans_clustering(
        self,
        df: pd.DataFrame,
        n_clusters: int = 5,
        features: List[str] = None
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        try:
            if features is None:
                features = ["vendor_name", "account_descr", "dept_name"]
            
            feature_df = df[features + ["monetary_amount"]].copy()
            
            label_encoders = {}
            for feature in features:
                if feature_df[feature].dtype == 'object':
                    le = LabelEncoder()
                    feature_df[feature] = le.fit_transform(feature_df[feature].fillna("Unknown"))
                    label_encoders[feature] = le
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(feature_df)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            
            model_info = {
                "n_clusters": n_clusters,
                "features_used": features,
                "silhouette_score": silhouette_avg,
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "inertia": kmeans.inertia_
            }
            
            model_path = os.path.join(self.models_path, f"kmeans_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            encoders_path = os.path.join(self.models_path, f"clustering_encoders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            scaler_path = os.path.join(self.models_path, f"clustering_scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            
            joblib.dump(kmeans, model_path)
            joblib.dump(label_encoders, encoders_path)
            joblib.dump(scaler, scaler_path)
            
            model_info.update({
                "model_path": model_path,
                "encoders_path": encoders_path,
                "scaler_path": scaler_path
            })
            
            return cluster_labels, silhouette_avg, model_info
            
        except Exception as e:
            raise MLModelException("KMeans Clustering", "training", {"error": str(e)})
    
    def train_expenditure_classifier(
        self,
        df: pd.DataFrame,
        threshold: float = 1000.0,
        features: List[str] = None
    ) -> Dict[str, Any]:
        try:
            if features is None:
                features = ["account_descr", "dept_name", "vendor_name"]
            
            df_clean = df[features + ["monetary_amount"]].copy().dropna()
            
            df_clean["expenditure_type"] = df_clean["monetary_amount"].apply(
                lambda x: "High Value" if x > threshold else "Low Value"
            )
            
            X = df_clean[features]
            y = df_clean["expenditure_type"]
            
            label_encoders = {}
            for feature in features:
                if X[feature].dtype == 'object':
                    le = LabelEncoder()
                    X[feature] = le.fit_transform(X[feature].fillna("Unknown"))
                    label_encoders[feature] = le
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(self.models_path, f"rf_classifier_{timestamp}.pkl")
            encoders_path = os.path.join(self.models_path, f"classifier_encoders_{timestamp}.pkl")
            
            joblib.dump(clf, model_path)
            joblib.dump(label_encoders, encoders_path)
            
            return {
                "model_type": "RandomForestClassifier",
                "accuracy": accuracy,
                "classification_report": report,
                "feature_importance": dict(zip(features, clf.feature_importances_)),
                "threshold": threshold,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "model_path": model_path,
                "encoders_path": encoders_path,
                "features_used": features
            }
            
        except Exception as e:
            raise MLModelException("RandomForest Classifier", "training", {"error": str(e)})
    
    async def predict_expenditure_type(
        self,
        model_path: str,
        encoders_path: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            clf = joblib.load(model_path)
            encoders = joblib.load(encoders_path)
            
            input_df = pd.DataFrame([input_data])
            
            for feature, encoder in encoders.items():
                if feature in input_df.columns:
                    try:
                        input_df[feature] = encoder.transform(input_df[feature].fillna("Unknown"))
                    except ValueError:
                        input_df[feature] = encoder.transform(["Unknown"])
            
            prediction = clf.predict(input_df)
            prediction_proba = clf.predict_proba(input_df)
            
            return {
                "prediction": prediction[0],
                "confidence": float(max(prediction_proba[0])),
                "probabilities": {
                    cls: float(prob) 
                    for cls, prob in zip(clf.classes_, prediction_proba[0])
                }
            }
            
        except Exception as e:
            raise MLModelException("RandomForest Classifier", "prediction", {"error": str(e)})
    
    async def save_model_metadata(
        self,
        session: AsyncSession,
        model_info: Dict[str, Any],
        model_type: str
    ) -> MLModelMetadata:
        try:
            metadata = MLModelMetadata(
                model_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_type=model_type,
                model_version="1.0.0",
                file_path=model_info.get("model_path", ""),
                accuracy_score=model_info.get("accuracy"),
                silhouette_score=model_info.get("silhouette_score"),
                training_data_size=model_info.get("training_samples", 0),
                features_used=json.dumps(model_info.get("features_used", [])),
                is_active=True,
                trained_at=datetime.utcnow()
            )
            
            await session.execute(
                select(MLModelMetadata)
                .where(MLModelMetadata.model_type == model_type)
                .update({"is_active": False})
            )
            
            session.add(metadata)
            await session.flush()
            await session.refresh(metadata)
            
            return metadata
            
        except Exception as e:
            raise MLModelException(model_type, "metadata saving", {"error": str(e)})

ml_service = MLService()