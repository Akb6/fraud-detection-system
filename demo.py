#!/usr/bin/env python3


import sys
import os
from fraud_detection_system import FraudDetectionSystem

def run_demo():
    """Run the complete fraud detection demo"""
    
    print("🚀 FRAUD DETECTION SYSTEM DEMO")
    print("=" * 60)
    print("This demo shows the complete fraud detection workflow")
    print("=" * 60)
    
    try:
        # Step 1: Generate synthetic data if needed
        if not os.path.exists('fraud_transactions.csv'):
            print("\n📊 Generating synthetic transaction data...")
            from generate_fraud_data import main as generate_data
            generate_data()
        
        # Step 2: Initialize the fraud detection system
        print("\n🔧 Initializing Fraud Detection System...")
        detector = FraudDetectionSystem()
        
        # Step 3: Load the data
        print("\n📁 Loading transaction data from multiple .pkl files...")
        if not detector.load_data_from_folder('data'):
         print("❌ Failed to load data from folder. Please check the 'data' folder.")
        return False

        
        # Step 4: Explore the data
        print("\n🔍 Performing exploratory data analysis...")
        detector.explore_data()
        
        # Step 5: Feature engineering
        print("\n⚙️ Engineering features for fraud detection...")
        detector.feature_engineering()
        
        # Step 6: Prepare features for ML
        print("\n🎯 Preparing features for machine learning...")
        detector.prepare_features()
        
        # Step 7: Train models
        print("\n🤖 Training machine learning models...")
        results = detector.train_models()
        
        # Step 8: Evaluate models
        print("\n📈 Evaluating model performance...")
        results_df, best_model = detector.evaluate_models()
        
        # Step 9: Generate visualizations
        print("\n📊 Generating performance visualizations...")
        detector.plot_results()
        
        # Step 10: Generate comprehensive report
        print("\n📋 Generating comprehensive fraud detection report...")
        report = detector.generate_report()
        
        # Save report to file
        with open('fraud_detection_report.txt', 'w') as f:
            f.write(report)
        
        print("\n✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Files generated:")
        print("  • fraud_transactions.csv - Synthetic transaction dataset")
        print("  • fraud_detection_results.png - Performance visualizations")
        print("  • fraud_detection_report.txt - Comprehensive analysis report")
        print("\n🎉 The fraud detection system is ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("Please check your Python environment and dependencies.")
        return False

def show_usage_examples():
    """Show additional usage examples"""
    
    print("\n🔍 ADDITIONAL USAGE EXAMPLES")
    print("=" * 40)
    
    print("# Example 1: Score individual transactions")
    print("detector = FraudDetectionSystem()")
    print("detector.load_data('your_data.csv')")
    print("detector.train_models()")
    print("best_model = detector.models['XGBoost']")
    print("fraud_score = best_model.predict_proba([features])[0][1]")
    print()
    print("# Example 2: Batch processing")
    print("batch_transactions = pd.read_csv('new_transactions.csv')")
    print("predictions = detector.predict_batch(batch_transactions)")
    print()
    print("# Example 3: Custom feature engineering")
    print("detector.df['custom_feature'] = detector.df['TX_AMOUNT'] / detector.df['customer_mean']")

if __name__ == "__main__":
    print("Starting Fraud Detection System Demo...")
    
    # Run the main demo
    success = run_demo()
    
    if success:
        # Show additional examples
        show_usage_examples()
        
        print("\n🎓 NEXT STEPS:")
        print("1. Review the generated report: fraud_detection_report.txt")
        print("2. Examine the visualizations: fraud_detection_results.png")
        print("3. Customize the models for your specific use case")
        print("4. Deploy the system for real-time fraud detection")
        print("\nHappy fraud hunting! 🕵️‍♂️")
    
    else:
        print("\n💡 TROUBLESHOOTING:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Verify data file exists and is readable")
        print("4. Check for any import errors")
