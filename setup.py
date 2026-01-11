"""
Setup script untuk initialize database dan folders
Run this first: python setup.py
"""
import os
import database

print("="*60)
print("SKIN CANCER DETECTION - SETUP")
print("="*60)

# Create necessary folders
folders = [
    'uploads',
    'uploads/pending',
    'uploads/reviewed',
    'templates',
    'model'
]

print("\n1. Creating folders...")
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"   ✓ {folder}/")

# Initialize database
print("\n2. Initializing database...")
database.init_db()

print("\n" + "="*60)
print("SETUP COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Place your model files in 'model/' folder:")
print("   - model.tflite")
print("   - labels.txt")
print("\n2. Create HTML templates in 'templates/' folder:")
print("   - admin_login.html")
print("   - admin_dashboard.html")
print("   - admin_review.html")
print("\n3. Run the app:")
print("   python app.py")
print("\n4. Access admin panel:")
print("   http://localhost:5000/admin")
print("   Default login: admin / admin123")
print("="*60)