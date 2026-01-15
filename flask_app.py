from flask import Flask, render_template, request, jsonify
import app
import os

flask_app = Flask(__name__)

# Global variables to store loaded data
df = None
tfidf = None
tfidf_matrix = None
model = None
embeddings = None

# Initialize the system on startup
@flask_app.before_request
def initialize():
    global df, tfidf, tfidf_matrix, model, embeddings
    if df is None:
        print("🔄 Loading recommendation system...")
        # NOTE: Make sure this path points to your actual CSV file
        data_path = "indian_dishes_dataset.csv" 

        if not os.path.exists(data_path):
            print(f"❌ Dataset not found at {data_path}")
            # Ensure proper handling if the dataset isn't there
            return

        # Pass the global variables to the setup function in app.py
        df, (tfidf, tfidf_matrix, model, embeddings) = app.build_system(data_path)
        print("✅ System ready!")


@flask_app.route('/')
def home():
    # Assuming 'index.html' is in the 'templates' folder
    return render_template('index.html', flavors=list(app.flavor_dict.keys()))


@flask_app.route('/api/dishes', methods=['GET'])
def get_dishes():
    """Get all dish names for autocomplete"""
    if df is None:
        return jsonify({"error": "System not initialized"}), 500

    dishes = df["name"].tolist()
    return jsonify({"dishes": dishes})


@flask_app.route('/api/recommend/dish', methods=['POST'])
def recommend_by_dish():
    """Recommend based on dish name - WITHOUT generating recipe steps"""
    if df is None:
        return jsonify({"error": "System not initialized"}), 500

    data = request.get_json()
    dish_name = data.get('dish_name', '')
    top_n = data.get('top_n', 5)
    dietary_filters = data.get('dietary_filters', [])

    if not dish_name:
        return jsonify({"error": "Dish name is required"}), 400

    result = app.recommend(
        dish_name, df, tfidf, tfidf_matrix, model, embeddings,
        dietary_filters=dietary_filters,
        top_n=top_n,
        generate_steps=False 
    )

    if result.empty:
        return jsonify({"error": f"Dish '{dish_name}' not found"}), 404

    recommendations = []
    for idx, row in result.iterrows():
        recommendation = {
            "name": row["name"],
            "cuisine": row["cuisine"],
            "ingredients": row["ingredients"],
            "difficulty": row["difficulty"],
            "rating": float(row["rating"]),
            "flavor_profile": dict(row["flavor_profile"]),
            "score": float(row["score"]),
            "has_recipe": False 
        }
        
        for flag in ["is_vegetarian", "is_vegan", "is_dairy_free", "is_gluten_free"]:
            if flag in row:
                recommendation[flag] = bool(row[flag])
                
        recommendations.append(recommendation)

    return jsonify({"recommendations": recommendations})


@flask_app.route('/api/recommend/ingredients', methods=['POST'])
def recommend_by_ingredients_route():
    """Recommend based on ingredients and flavor - WITHOUT generating recipe steps"""
    if df is None:
        return jsonify({"error": "System not initialized"}), 500

    data = request.get_json()
    ingredients = data.get('ingredients', '')
    flavor = data.get('flavor', '')
    exclude_ingredients = data.get('exclude_ingredients', '')
    dietary_filters = data.get('dietary_filters', [])
    top_n = data.get('top_n', 5)

    if not ingredients:
        return jsonify({"error": "Ingredients are required"}), 400

    result = app.recommend_by_ingredients(
        ingredients, flavor, df, tfidf, tfidf_matrix, model, embeddings,
        exclude_ingredients=exclude_ingredients,
        dietary_filters=dietary_filters,
        top_n=top_n,
        generate_steps=False  
    )

    if result.empty:
        return jsonify({"recommendations": [], "message": "No dishes found matching your criteria"}), 200

    recommendations = []
    for idx, row in result.iterrows():
        recommendation = {
            "name": row["name"],
            "cuisine": row["cuisine"],
            "ingredients": row["ingredients"],
            "difficulty": row["difficulty"],
            "rating": float(row["rating"]),
            "flavor_profile": dict(row["flavor_profile"]),
            "score": float(row["score"]),
            "has_recipe": False
        }
        
        for flag in ["is_vegetarian", "is_vegan", "is_dairy_free", "is_gluten_free"]:
            if flag in row:
                recommendation[flag] = bool(row[flag])
                
        recommendations.append(recommendation)

    return jsonify({"recommendations": recommendations})


@flask_app.route('/api/recipe/<path:dish_name>', methods=['POST'])
def get_recipe_for_dish(dish_name):
    """Generate recipe steps for a specific dish when user clicks on it"""
    if df is None:
        return jsonify({"error": "System not initialized"}), 500

    data = request.get_json()
    ingredients = data.get('ingredients', '')
    
    if not dish_name:
        return jsonify({"error": "Dish name is required"}), 400
    
    if not ingredients:
        dish_row = df[df["name"] == dish_name]
        if not dish_row.empty:
            ingredients = dish_row.iloc[0]["ingredients"]
        else:
            return jsonify({"error": "Ingredients not found for this dish"}), 404
    
    try:
        print(f"🔄 Generating recipe for: {dish_name}")
        steps = app.get_recipe_steps(dish_name, ingredients)
        
        # 🌟 FIX: Perform final cleanup on the backend before sending to frontend JS
        # This replaces the initial junk line breaks and ensures the steps start cleanly
        cleaned_steps = steps.replace('\n\n\n', '\n\n').strip() 
        
        return jsonify({
            "dish_name": dish_name,
            "ingredients": ingredients,
            "steps": cleaned_steps,
            "success": True
        })
    except Exception as e:
        print(f"❌ Error generating recipe: {str(e)}")
        return jsonify({
            "error": f"Failed to generate recipe: {str(e)}",
            "success": False
        }), 500


if __name__ == '__main__':
    flask_app.run(debug=True, host='0.0.0.0', port=5000)