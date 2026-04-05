"""Machine Learning-Based Credit Card Fraud Detection
    Using Logistic Regression and K-Nearest Neighbors"""

    # ─────────────────────────────────────────────
    # 1. IMPORTS
    # ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, f1_score
    )


    # ─────────────────────────────────────────────
    # 2. CREATE SYNTHETIC DATASET
    # ─────────────────────────────────────────────
np.random.seed(42)
n = 1000

    # Generate legitimate transactions
n_legit = 970
legit = pd.DataFrame({
    'amount':          np.random.exponential(scale=80,  size=n_legit),
    'location_risk':   np.random.uniform(0.0, 0.5,      size=n_legit),
    'txn_frequency':   np.random.poisson(lam=6,         size=n_legit),
    'hour_of_day':     np.random.randint(7, 22,          size=n_legit),
    'is_fraud':        0
    })

    # Generate fraudulent transactions (distinct pattern)
n_fraud = 30
fraud = pd.DataFrame({
    'amount':          np.random.exponential(scale=400, size=n_fraud),
    'location_risk':   np.random.uniform(0.7, 1.0,      size=n_fraud),
    'txn_frequency':   np.random.poisson(lam=1,         size=n_fraud),
    'hour_of_day':     np.random.randint(0, 6,           size=n_fraud),
    'is_fraud':        1
    })
if __name__ == "__main__":
    data = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)

    print("=" * 55)
    print("  FRAUD DETECTION — DATASET OVERVIEW")
    print("=" * 55)
    print(data.head(8).to_string(index=False))
    print(f"\nShape: {data.shape}")
    print(f"\nClass distribution:\n{data['is_fraud'].value_counts().rename({0:'Legitimate', 1:'Fraudulent'})}")
    print(f"\nMissing values: {data.isnull().sum().sum()}")


    # ─────────────────────────────────────────────
    # 3. PREPROCESSING
    # ─────────────────────────────────────────────
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit + transform on train
    X_test_sc  = scaler.transform(X_test)        # transform only on test

    print(f"\nTrain size : {X_train_sc.shape[0]} samples")
    print(f"Test  size : {X_test_sc.shape[0]}  samples")


    # ─────────────────────────────────────────────
    # 4. TRAIN LOGISTIC REGRESSION
    # ─────────────────────────────────────────────
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_sc, y_train)
    y_pred_lr   = lr.predict(X_test_sc)
    y_prob_lr   = lr.predict_proba(X_test_sc)[:, 1]

    acc_lr  = accuracy_score(y_test, y_pred_lr)
    f1_lr   = f1_score(y_test, y_pred_lr, zero_division=0)
    auc_lr  = roc_auc_score(y_test, y_prob_lr)

    print("\n" + "=" * 55)
    print("  LOGISTIC REGRESSION RESULTS")
    print("=" * 55)
    print(f"Accuracy  : {acc_lr:.4f}")
    print(f"F1 Score  : {f1_lr:.4f}")
    print(f"ROC-AUC   : {auc_lr:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr,
    target_names=['Legitimate', 'Fraudulent'], zero_division=0))


    # ─────────────────────────────────────────────
    # 5. TUNE & TRAIN KNN
    # ─────────────────────────────────────────────
    # Find best K using GridSearchCV
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 15]}
    grid_search = GridSearchCV(
    KNeighborsClassifier(), param_grid,
    cv=5, scoring='f1', error_score=0
    )
    grid_search.fit(X_train_sc, y_train)
    best_k = grid_search.best_params_['n_neighbors']

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_sc, y_train)
    y_pred_knn  = knn.predict(X_test_sc)
    y_prob_knn  = knn.predict_proba(X_test_sc)[:, 1]

    acc_knn = accuracy_score(y_test, y_pred_knn)
    f1_knn  = f1_score(y_test, y_pred_knn, zero_division=0)
    auc_knn = roc_auc_score(y_test, y_prob_knn)

    print("\n" + "=" * 55)
    print(f"  KNN RESULTS  (best K = {best_k})")
    print("=" * 55)
    print(f"Accuracy  : {acc_knn:.4f}")
    print(f"F1 Score  : {f1_knn:.4f}")
    print(f"ROC-AUC   : {auc_knn:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_knn,
    target_names=['Legitimate', 'Fraudulent'], zero_division=0))

    # K vs F1 scores for all tested values
    k_values = param_grid['n_neighbors']
    k_f1s    = grid_search.cv_results_['mean_test_score']


    # ─────────────────────────────────────────────
    # 6. MODEL COMPARISON SUMMARY
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 55)
    print(f"{'Metric':<20} {'Logistic Reg':>14} {'KNN':>10}")
    print("-" * 45)
    print(f"{'Accuracy':<20} {acc_lr:>14.4f} {acc_knn:>10.4f}")
    print(f"{'F1 Score':<20} {f1_lr:>14.4f} {f1_knn:>10.4f}")
    print(f"{'ROC-AUC':<20} {auc_lr:>14.4f} {auc_knn:>10.4f}")
    best_model_name = "Logistic Regression" if auc_lr >= auc_knn else f"KNN (K={best_k})"
    print(f"\nBest model by AUC: {best_model_name}")


    # ─────────────────────────────────────────────
    # 7. VISUALIZATIONS
    # ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Fraud Detection — Model Performance Dashboard", fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {'lr': '#534AB7', 'knn': '#0F6E56', 'fraud': '#D85A30', 'legit': '#1D9E75'}

    # — Plot 1: Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(['Logistic\nRegression', f'KNN\n(K={best_k})'],
            [acc_lr, acc_knn],
            color=[colors['lr'], colors['knn']], width=0.5, edgecolor='white')
    ax1.set_ylim(0.85, 1.0)
    ax1.set_title('Accuracy Comparison', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    for bar, val in zip(bars, [acc_lr, acc_knn]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.spines[['top','right']].set_visible(False)

    # — Plot 2: F1 and AUC grouped bar
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(2)
    w = 0.3
    ax2.bar(x - w/2, [f1_lr, f1_knn],   width=w, label='F1 Score', color=colors['lr'],   edgecolor='white')
    ax2.bar(x + w/2, [auc_lr, auc_knn], width=w, label='ROC-AUC',  color=colors['knn'],  edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Logistic Reg', f'KNN K={best_k}'])
    ax2.set_ylim(0, 1.1)
    ax2.set_title('F1 Score & ROC-AUC', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines[['top','right']].set_visible(False)

    # — Plot 3: K vs F1 line
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(k_values, k_f1s, marker='o', color=colors['knn'], linewidth=2, markersize=7)
    ax3.axvline(best_k, color=colors['fraud'], linestyle='--', linewidth=1.5, label=f'Best K={best_k}')
    ax3.set_xlabel('K Value')
    ax3.set_ylabel('Mean F1 (CV=5)')
    ax3.set_title('KNN: K vs F1 Score', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.spines[['top','right']].set_visible(False)

    # — Plot 4: Confusion matrix — LR
    ax4 = fig.add_subplot(gs[1, 0])
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    disp = ConfusionMatrixDisplay(cm_lr, display_labels=['Legit', 'Fraud'])
    disp.plot(ax=ax4, colorbar=False, cmap='Purples')
    ax4.set_title('Confusion Matrix — LR', fontweight='bold')

    # — Plot 5: Confusion matrix — KNN
    ax5 = fig.add_subplot(gs[1, 1])
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    disp2 = ConfusionMatrixDisplay(cm_knn, display_labels=['Legit', 'Fraud'])
    disp2.plot(ax=ax5, colorbar=False, cmap='Greens')
    ax5.set_title(f'Confusion Matrix — KNN (K={best_k})', fontweight='bold')

    # — Plot 6: ROC curves
    ax6 = fig.add_subplot(gs[1, 2])
    fpr_lr,  tpr_lr,  _ = roc_curve(y_test, y_prob_lr)
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
    ax6.plot(fpr_lr,  tpr_lr,  color=colors['lr'],  lw=2, label=f'LR  (AUC={auc_lr:.3f})')
    ax6.plot(fpr_knn, tpr_knn, color=colors['knn'], lw=2, label=f'KNN (AUC={auc_knn:.3f})')
    ax6.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4)
    ax6.set_xlabel('False Positive Rate')
    ax6.set_ylabel('True Positive Rate')
    ax6.set_title('ROC Curves', fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.spines[['top','right']].set_visible(False)

    plt.savefig('fraud_detection_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved → fraud_detection_results.png")
    plt.show()


    # ─────────────────────────────────────────────
    # 8. PREDICT NEW TRANSACTIONS
    # ─────────────────────────────────────────────

def predict_transaction(amount, location_risk, txn_frequency, hour_of_day):
    features = np.array([[amount, location_risk, txn_frequency, hour_of_day]])
    features_sc = scaler.transform(features)

    # Use ONE model (logistic regression)
    pred = lr.predict(features_sc)[0]
    prob = lr.predict_proba(features_sc)[0][1]

    risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
    label = "FRAUDULENT" if pred == 1 else "Legitimate"

    return {
        'label': label,
        'fraud_probability': round(prob, 4),
        'risk_level': risk
    }
    """
    Predict if a transaction is fraudulent.

    Parameters:
        amount          : Transaction amount in dollars
        location_risk   : Risk score of transaction location (0.0 – 1.0)
        txn_frequency   : Number of recent transactions by this card
        hour_of_day     : Hour of day (0–23)

    Returns:
        dict with label, probability, and risk level
    """
    features = np.array([[amount, location_risk, txn_frequency, hour_of_day]])
    features_sc = scaler.transform(features)
    pred  = model.predict(features_sc)[0]
    prob  = model.predict_proba(features_sc)[0][1]
    risk  = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
    label = "FRAUDULENT" if pred == 1 else "Legitimate"
    return {'label': label, 'fraud_probability': round(prob, 4), 'risk_level': risk}


    print("\n" + "=" * 55)
    print("  LIVE TRANSACTION PREDICTIONS")
    print("=" * 55)

    test_transactions = [
    {"amount": 4800, "location_risk": 0.92, "txn_frequency": 1, "hour_of_day": 2,  "note": "Suspicious — large amount, odd hour"},
    {"amount": 52,   "location_risk": 0.08, "txn_frequency": 7, "hour_of_day": 13, "note": "Normal — small, low risk"},
    {"amount": 310,  "location_risk": 0.55, "txn_frequency": 3, "hour_of_day": 21, "note": "Borderline — moderate signals"},
    {"amount": 9999, "location_risk": 0.98, "txn_frequency": 1, "hour_of_day": 4,  "note": "Very suspicious"},
    ]

    for i, txn in enumerate(test_transactions, 1):
        result = predict_transaction(
        txn['amount'], txn['location_risk'],
        txn['txn_frequency'], txn['hour_of_day']
    )
    print(f"\nTransaction {i}: {txn['note']}")
    print(f"  Amount: ${txn['amount']:,.0f}  |  Location risk: {txn['location_risk']}  |  Hour: {txn['hour_of_day']:02d}:00")
    print(f"  → Result       : {result['label']}")
    print(f"  → Fraud prob   : {result['fraud_probability']:.2%}")
    print(f"  → Risk level   : {result['risk_level']}")

    print("\n" + "=" * 55)
    print("  DONE. Model ready for use.")
    print("=" * 55)
