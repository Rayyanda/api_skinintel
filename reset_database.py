"""
Reset database - WARNING: This will delete all data!
Run this: python reset_database.py
"""
import os
import database

print("=" * 60)
print("DATABASE RESET SCRIPT")
print("=" * 60)
print("\n⚠️  WARNING: This will DELETE all existing data!")
print("Including:")
print("  - All user accounts")
print("  - All uploads")
print("  - All admin accounts")
print()

confirm = input("Type 'YES' to confirm: ")

if confirm != 'YES':
    print("Cancelled.")
    exit()

# Delete old database
if os.path.exists('database.db'):
    os.remove('database.db')
    print("\n✓ Old database deleted")

# Create new database with updated schema
database.init_db()

print("\n" + "=" * 60)
print("DATABASE RESET COMPLETE!")
print("=" * 60)
print("\nDefault admin account created:")
print("  Username: admin")
print("  Password: admin123")
print("\nYou can now:")
print("1. Register new user accounts via app")
print("2. Run the server: python app.py")
print("=" * 60)