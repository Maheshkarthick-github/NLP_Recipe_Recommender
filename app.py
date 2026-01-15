import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
import time
import requests 
import json 
import os 

# Import the new Google GenAI library
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️ google-genai library not installed. Run: pip install google-genai")

# -------------------- Configure Gemini --------------------
# NOTE: Replace "AIzaSyAaaxAk-N90tXYwRj-c7UiNbyKtog66vbc" with your actual Gemini API Key
GEMINI_API_KEY = "AIzaSyAaaxAk-N90tXYwRj-c7UiNbyKtog66vbc" 

# Initialize Gemini client
gemini_client = None
if GENAI_AVAILABLE:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"⚠️ Failed to initialize Gemini client: {e}")

# -------------------- Global Weights & Keywords (MODIFIED) --------------------

NON_VEG_KEYWORDS = ['meat', 'chicken', 'mutton', 'beef', 'pork', 'fish', 'prawn', 'shrimp', 'seafood', 'lamb', 'bacon', 'ham']

# Define importance weights (Carb Base increased to 5.0)
IMPORTANCE_WEIGHTS = {
    'protein': 3.5,     # High priority
    'carb_base': 5.0,   # 🌟 INCREASED TO 5.0: This ensures rice/flour dishes dominate when listed.
    'dairy_base': 1.5,  # Medium priority
    'default': 1.0 
}

# Map ingredients to their importance category (using lemmatized forms)
IMPORTANCE_CATEGORIES = {
    # Proteins (Primary focus for Non-Veg filter/recommendation)
    'chicken': 'protein', 'mutton': 'protein', 'lamb': 'protein', 'beef': 'protein', 'pork': 'protein', 'fish': 'protein', 
    'prawn': 'protein', 'shrimp': 'protein', 'paneer': 'protein', 'tofu': 'protein', 'egg': 'protein',

    # Base Carbs (Rice, Wheat/Flour)
    'rice': 'carb_base', 'flour': 'carb_base', 'atta': 'carb_base', 'maida': 'carb_base', 'semolina': 'carb_base',

    # Base Dairy/Cream (Used as main component in gravies/curries)
    'yogurt': 'dairy_base', 'milk': 'dairy_base', 'cream': 'dairy_base', 'curd': 'dairy_base',
}
# ----------------------------------------------------------------------------------

def verify_gemini_api():
    """Verify that Gemini API is accessible"""
    global gemini_client
    
    if not GENAI_AVAILABLE or gemini_client is None:
        print("❌ Gemini client not available - will use template recipes")
        return False
    
    try:
        print("\n🔍 Testing Gemini API connection...")
        
        models_to_test = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
        
        for model_name in models_to_test:
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents="Say 'API Working' in exactly 2 words"
                )
                
                if response and response.text:
                    print(f"✅ {model_name} is accessible and working!")
                    print(f"   Test response: {response.text}")
                    return True
                    
            except Exception as e:
                error_msg = str(e)
                if "not found" in error_msg.lower() or "404" in error_msg:
                    print(f"⚠️ {model_name}: Model not found")
                else:
                    print(f"⚠️ {model_name}: {error_msg[:100]}")
        
        print("❌ No working Gemini models found - will use template recipes")
        return False
        
    except Exception as e:
        print(f"❌ API verification failed: {e}")
        return False

# -------------------- GEMINI FUNCTION (SDK-Based, Structured Output) --------------------
def generate_recipe_with_gemini(dish_name, ingredients, max_retries=2):
    """
    Generate recipe steps using Gemini AI via the SDK with a clean, highly structured text format.
    """
    global gemini_client
    
    if gemini_client is None:
        return generate_template_steps(dish_name, ingredients)

    # MODIFIED PROMPT: Uses '@@@' delimiter and removes voice placeholder.
    prompt = f"""You are an expert Indian chef. Generate a detailed, highly structured recipe for {dish_name}.

**Your primary available ingredients are:** {ingredients}
**You can assume the user has common essentials like:** Salt, Water, and Cooking Oil/Ghee.

**STRICT FORMATTING REQUIREMENTS:**
1.  **Recipe Title:** Output the title on a single line: 'RECIPE TITLE: {dish_name.upper()}'\n\n
2.  **Ingredients Section:** Output the header 'INGREDIENTS:' on its own line. On the next line, list ALL ingredients (provided + assumed), separated by commas. Follow this with a double newline.\n\n
3.  **Preparation Steps:** Output the header 'PREPARATION STEPS:' on its own line, followed by a double newline. Then, use numbered steps. IMPORTANT: START EACH NEW STEP WITH THE SEQUENCE '@@@' followed by the number and a period (e.g., '@@@1. Prepare...').
4.  Do NOT include any lines related to a 'Voice Assistant' or 'Click for help'.
"""

    models_to_try = [
        "gemini-2.5-flash", 
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"  Attempting to generate recipe using {model_name}...")
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": 0.7, 
                    "max_output_tokens": 2048,
                    "top_k": 40,
                    "top_p": 0.95
                }
            )
            
            recipe_text = response.text.strip()
            
            # Validate that we got a substantial response
            if len(recipe_text) > 100 and "RECIPE" in recipe_text[:50].upper():
                print(f"✅ Generated recipe for {dish_name} using {model_name}")
                return recipe_text
            
            print(f"⚠️ Response from {model_name} was too short or invalid, trying next model...")

        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower():
                print(f"⚠️ {model_name} Quota/Rate Limit Error. Trying next model...")
            elif "not found" in error_msg.lower() or "404" in error_msg:
                 print(f"⚠️ Model {model_name} not found, trying next model...")
            else:
                print(f"⚠️ SDK Error with {model_name} for {dish_name}: {error_msg[:100]}. Trying next model...")
            time.sleep(1) 

    print(f"⚠️ All Gemini attempts failed for {dish_name}, using template")
    return generate_template_steps(dish_name, ingredients)

# -------------------- TEMPLATE FALLBACK FUNCTION (Structured Output) --------------------
def generate_template_steps(dish_name, ingredients):
    """Generate template-based recipe steps as highly structured plain text fallback."""
    dish_lower = dish_name.lower()
    
    ingredients_provided = [item.strip().title() for item in ingredients.split(',') if item.strip()]
    all_ingredients = ", ".join(ingredients_provided + ["Salt", "Water", "Cooking Oil / Ghee"])
    
    output = f"RECIPE TITLE: {dish_name.upper()}\n\n"
    
    output += "INGREDIENTS: "
    output += all_ingredients
    output += "\n\n"
    
    output += "PREPARATION STEPS:\n\n"
    
    main_ing = ingredients_provided[0] if ingredients_provided else "your main ingredient"
    
    if 'curry' in dish_lower or 'masala' in dish_lower:
        steps = f"""
@@@1. Prepare your main ingredients (e.g., chop {main_ing} and other available vegetables/protein).
@@@2. Heat Oil/Ghee and temper with any available whole spices.
@@@3. Sauté available aromatics (like onion, ginger, garlic, if listed) until fragrant.
@@@4. Add available chopped tomatoes and cook until softened.
@@@5. Stir in available spice powders (turmeric, chili, coriander, etc.) and Salt. Mix well.
@@@6. Add the prepared main ingredients and a small amount of Water. Cover and simmer until cooked through (20-30 mins).
@@@7. Adjust seasoning and serve hot.
"""
    
    elif 'biryani' in dish_lower or 'pulao' in dish_lower:
        steps = f"""
@@@1. If rice is available, soak it using Water. Marinate available {main_ing} using any available yogurt and spices (plus Salt).
@@@2. Heat Oil/Ghee in a pot. Fry available onion slices until very brown.
@@@3. Add marinated ingredients and cook for 5-7 mins.
@@@4. Par-boil the rice (if available) using Salt and Water and drain it.
@@@5. Layer the par-cooked rice over the cooked ingredients.
@@@6. Cover tightly and cook on the lowest heat (Dum) for 25-30 minutes.
@@@7. Gently mix and serve.
"""

    elif 'pakora' in dish_lower or 'bhaji' in dish_lower or 'vada' in dish_lower or 'fritter' in dish_lower:
        steps = """
@@@1. Prepare vegetables/main ingredients (slice thin). If flour is available, mix it with Water and available spices/Salt to create a thick batter.
@@@2. Heat Oil/Ghee for frying in a deep pan.
@@@3. Dip the ingredients in the batter (if batter was made).
@@@4. Carefully fry small batches in the hot Oil until golden brown and crispy.
@@@5. Remove, drain, and serve hot.
"""
    
    else:
        steps = f"""
@@@1. Prepare your available ingredients (wash, chop {main_ing} and others).
@@@2. Heat Oil/Ghee and add any available whole spices for tempering.
@@@3. Sauté available aromatics (like onion, ginger, garlic, if listed) until soft.
@@@4. Add the rest of your {main_ing}/vegetables and any available spice powders (plus Salt). Mix well.
@@@5. Cover and cook on medium-low heat, adding Water if necessary, until all ingredients are tender (10-15 mins).
@@@6. Adjust seasoning and serve hot.
"""
    
    # Clean up the initial newline and strip trailing whitespace
    output += steps.strip()
    
    return output

# -------------------- Load & Clean --------------------

def load_data(file_path):
    """
    Loads data from CSV or JSON file, using robust settings for CSV
    to handle potential tokenization errors.
    """
    if file_path.endswith(".csv"):
        try:
            # Attempt 1: Standard, fast C engine 
            df = pd.read_csv(file_path)
        except pd.errors.ParserError as e:
            print(f"⚠️ ParserError encountered: {e}. Attempting robust CSV load...")
            # Attempt 2: Use Python engine and skip bad lines for robustness
            df = pd.read_csv(file_path, engine='python', on_bad_lines='skip') 
    else:
        df = pd.read_json(file_path)
        
    print(f"✅ Loaded {len(df)} dishes")
    return df

# 🌟 NEW Helper function for basic spell correction (Must be defined outside clean_and_lemmatize)
def simple_spell_correction(token):
    """Applies basic correction for common ingredient misspellings."""
    if 'tomat' in token:
        return 'tomato'
    if 'onions' in token or 'onionz' in token:
        return 'onion'
    # Add other common misspellings here if needed
    return token

def clean_and_lemmatize(df):
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    lemmatizer = WordNetLemmatizer()

    # 🌟 MODIFIED process function to use the external spell correction helper
    def process(ingredients):
        ingredients = str(ingredients).lower()
        ingredients = re.sub(r"[^a-zA-Z, ]", "", ingredients)
        
        # 1. Split into raw tokens
        raw_tokens = [tok.strip() for tok in ingredients.split(",") if tok.strip()]
        
        # 2. Apply spell correction to tokens (using the function defined above)
        corrected_tokens = [simple_spell_correction(tok) for tok in raw_tokens]
        
        # 3. Apply lemmatization
        tokens = [lemmatizer.lemmatize(tok) for tok in corrected_tokens]
        return " ".join(tokens)

    df["clean_ingredients"] = df["ingredients"].apply(process)
    return df


# -------------------- Flavor Profiles --------------------

flavor_dict = {
    "spicy": ["chili", "pepper", "ginger", "garlic", "cayenne", "jalapeno"],
    "sweet": ["sugar", "jaggery", "honey", "raisins", "date", "syrup"],
    "sour": ["tamarind", "lemon", "vinegar", "yogurt", "lime", "kokum"],
    "salty": ["salt", "soy sauce", "pickle"],
    "umami": ["mushroom", "soy", "paneer", "tomato", "cheese"],
    "fresh": ["coriander", "mint", "basil", "parsley", "cilantro"],
    "creamy": ["milk", "cream", "cashew", "yogurt", "coconut milk"],
    "smoky": ["smoked", "roasted", "charcoal", "tandoor", "bbq"]
}


def compute_flavor_profile(df):
    profiles = []
    for ing in df["clean_ingredients"]:
        tokens = ing.split()
        profile = defaultdict(int)
        for flavor, keywords in flavor_dict.items():
            for k in keywords:
                if k in tokens:
                    profile[flavor] += 1
        profiles.append(profile)
    df["flavor_profile"] = profiles
    return df


# -------------------- Dietary Classification --------------------

def classify_dietary_restrictions(df):
    """Classify dishes based on ingredients for dietary restrictions"""
    # NOTE: NON_VEG_KEYWORDS is defined globally, using it here for consistency
    non_veg_keywords = NON_VEG_KEYWORDS
    egg_keywords = ['egg']
    dairy_keywords = ['milk', 'cream', 'cheese', 'butter', 'paneer', 'yogurt', 'curd', 'ghee', 'lassi']
    gluten_keywords = ['wheat', 'barley', 'rye', 'atta', 'maida', 'flour', 'semolina', 'rava', 'sooji', 'bread']
    
    df['is_vegetarian'] = False
    df['is_vegan'] = False
    df['is_dairy_free'] = False
    df['is_gluten_free'] = False
    
    for idx, row in df.iterrows():
        ingredients = row['clean_ingredients'].split()
        
        has_non_veg = any(keyword in ingredients for keyword in non_veg_keywords)
        has_egg = any(keyword in ingredients for keyword in egg_keywords)
        has_dairy = any(keyword in ingredients for keyword in dairy_keywords)
        has_gluten = any(keyword in ingredients for keyword in gluten_keywords)
        
        if not has_non_veg:
            df.at[idx, 'is_vegetarian'] = True
            if not has_egg and not has_dairy:
                df.at[idx, 'is_vegan'] = True
        
        if not has_dairy:
            df.at[idx, 'is_dairy_free'] = True
            
        if not has_gluten:
            df.at[idx, 'is_gluten_free'] = True
    
    return df


# -------------------- Vectorization --------------------

def build_tfidf(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df["clean_ingredients"])
    return tfidf, tfidf_matrix


def build_embeddings(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["clean_ingredients"].tolist(), convert_to_tensor=True)
    return model, embeddings


# -------------------- Similarity Metrics --------------------

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0


def flavor_match(profile1, profile2):
    common = set(profile1.keys()) & set(profile2.keys())
    if not common:
        return 0
    score = sum(min(profile1[f], profile2[f]) for f in common)
    return score / (sum(profile1.values()) + 1e-5)


def get_ingredient_weight(ingredient, lemmatizer):
    """Retrieves the importance weight for a given ingredient."""
    # Ensure ingredient is lemmatized for accurate lookup
    lemmatized_ing = lemmatizer.lemmatize(ingredient.strip().lower())
    
    category = IMPORTANCE_CATEGORIES.get(lemmatized_ing)
    if category:
        return IMPORTANCE_WEIGHTS[category]
    return IMPORTANCE_WEIGHTS['default']


# 🌟 MODIFIED FUNCTION: Uses weighted coverage
def compute_ingredient_coverage(user_ingredients_set, dish_ingredients_str):
    """Calculate the weighted coverage score."""
    if not dish_ingredients_str:
        return 0
    
    lemmatizer = WordNetLemmatizer()
    
    # Split the raw ingredients string by comma and clean
    dish_ingredients_list = [ing.strip() for ing in dish_ingredients_str.split(',') if ing.strip()]
    
    total_weight = 0.0
    matched_weight = 0.0
    
    # Get the set of lemmatized dish ingredients for quick lookup
    # lemmatized_dish_set = set([lemmatizer.lemmatize(ing) for ing in dish_ingredients_list]) # Not needed here
    
    for dish_ing_raw in dish_ingredients_list:
        weight = get_ingredient_weight(dish_ing_raw, lemmatizer)
        total_weight += weight
        
        # Check if the dish ingredient is in the user's matched set (which is already lemmatized)
        lemmatized_dish_ing = lemmatizer.lemmatize(dish_ing_raw)
        
        if lemmatized_dish_ing in user_ingredients_set:
            matched_weight += weight
            
    if total_weight == 0:
        return 0
        
    return matched_weight / total_weight


def compute_weighted_score(cos_sim, emb_sim, jacc, flavor, rating, coverage=1.0, w=[0.15,0.15,0.35,0.1,0.05,0.2]):
    """Enhanced scoring with ingredient coverage"""
    return w[0]*cos_sim + w[1]*emb_sim + w[2]*jacc + w[3]*flavor + w[4]*(rating/5) + w[5]*coverage


# -------------------- Dietary Filters --------------------

def apply_dietary_filters(df, dietary_requirements):
    """Filter dataframe based on dietary requirements"""
    filtered_df = df.copy()
    initial_count = len(filtered_df)

    for requirement in dietary_requirements:
        if requirement == 'dairy-free' and 'is_dairy_free' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_dairy_free'] == True]
            print(f"  Applied dairy-free filter: {len(filtered_df)} dishes remaining")
        elif requirement == 'gluten-free' and 'is_gluten_free' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_gluten_free'] == True]
            print(f"  Applied gluten-free filter: {len(filtered_df)} dishes remaining")
        elif requirement == 'vegetarian' and 'is_vegetarian' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_vegetarian'] == True]
            print(f"  Applied vegetarian filter: {len(filtered_df)} dishes remaining")
        elif requirement == 'vegan' and 'is_vegan' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_vegan'] == True]
            print(f"  Applied vegan filter: {len(filtered_df)} dishes remaining")

    print(f"✅ Dietary filters applied: {initial_count} → {len(filtered_df)} dishes")
    return filtered_df


def contains_excluded_ingredient(dish_ingredients, exclude_ingredients):
    """Check if dish contains ANY of the excluded ingredients"""
    if not exclude_ingredients:
        return False

    exclude_str = str(exclude_ingredients).strip()
    if not exclude_str or exclude_str.lower() == "none":
        return False

    lemmatizer = WordNetLemmatizer()
    exclude_list = [lemmatizer.lemmatize(item.strip().lower()) for item in exclude_str.split(",") if item.strip()]
    if not exclude_list:
        return False

    dish_str = str(dish_ingredients).lower()
    dish_str = re.sub(r"[^a-zA-Z, ]", "", dish_str)
    dish_tokens = [lemmatizer.lemmatize(tok.strip()) for tok in dish_str.split(",") if tok.strip()]

    for excluded_item in exclude_list:
        for dish_token in dish_tokens:
            if excluded_item in dish_token.split() or excluded_item == dish_token:
                return True
    return False


def check_required_ingredients_present(dish_ingredients, user_ingredients_list):
    """Ensure user has main/protein ingredients for dish"""
    dish_lower = str(dish_ingredients).lower()

    if isinstance(user_ingredients_list, str):
        user_ingredients_list = [item.strip().lower() for item in user_ingredients_list.split(",")]
    else:
        user_ingredients_list = [str(item).strip().lower() for item in user_ingredients_list]

    proteins = {
        'chicken': ['chicken', 'hen'],
        'mutton': ['mutton', 'lamb', 'goat'],
        'fish': ['fish', 'salmon', 'tuna', 'mackerel', 'pomfret', 'hilsa', 'rohu'],
        'prawn': ['prawn', 'shrimp'],
        'beef': ['beef'],
        'pork': ['pork'],
        'egg': ['egg'],
        'paneer': ['paneer', 'cottage cheese'],
        'tofu': ['tofu']
    }

    user_proteins = set()
    for protein_type, keywords in proteins.items():
        if any(keyword in user_ingredients_list for keyword in keywords):
            user_proteins.add(protein_type)

    if user_proteins:
        dish_proteins = set()
        for protein_type, keywords in proteins.items():
            if any(keyword in dish_lower for keyword in keywords):
                dish_proteins.add(protein_type)
        if dish_proteins and not (dish_proteins & user_proteins):
            return False

    return True

def check_user_for_non_veg_preference(user_ingredients_str):
    """Checks if the user has provided any non-vegetarian ingredient."""
    user_ingredients_lower = [item.strip().lower() for item in user_ingredients_str.split(",") if item.strip()]
    
    # Use lemmatization for accurate check against common forms
    lemmatizer = WordNetLemmatizer()
    lemmatized_user_ingredients = [lemmatizer.lemmatize(ing) for ing in user_ingredients_lower]
    
    return any(keyword in lemmatized_user_ingredients for keyword in NON_VEG_KEYWORDS)

# -------------------- Get Steps and Recommend Functions --------------------

def get_recipe_steps(dish_name, ingredients):
    """
    Get recipe steps for a specific dish.
    This should be called separately when user wants to view a recipe.
    """
    print(f"\n🔄 Generating recipe for {dish_name}...")
    steps = generate_recipe_with_gemini(dish_name, ingredients)
    return steps


def recommend(dish_name, df, tfidf, tfidf_matrix, model, embeddings, dietary_filters=None, top_n=5, generate_steps=False):
    """
    Recommend similar dishes. Set generate_steps=True to generate recipes immediately.
    """
    if dish_name not in df["name"].values:
        print(f"❌ '{dish_name}' not found")
        return pd.DataFrame()

    working_df = df.copy()
    if dietary_filters:
        print(f"\n🔍 Applying dietary filters for dish recommendation: {dietary_filters}")
        working_df = apply_dietary_filters(working_df, dietary_filters)
        if len(working_df) == 0:
            print("❌ No dishes match the dietary requirements")
            return pd.DataFrame()

    idx = df[df["name"] == dish_name].index[0]

    cos_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    emb_sim = util.cos_sim(embeddings[idx], embeddings)[0].cpu().numpy()

    base_ingredients = set(df.loc[idx, "clean_ingredients"].split())
    base_flavor = df.loc[idx, "flavor_profile"]

    scores = []
    for i in working_df.index:
        if i == idx:
            continue
        dish_ingredients_set = set(df.loc[i, "clean_ingredients"].split())
        jacc = jaccard_similarity(base_ingredients, dish_ingredients_set)
        flav = flavor_match(base_flavor, df.loc[i, "flavor_profile"])
        rating = df.loc[i, "rating"] if "rating" in df.columns else 4.0
        # NOTE: Using standard (unweighted) coverage here as the base recipe is the source dish
        coverage = compute_ingredient_coverage(base_ingredients, df.loc[i, "ingredients"]) 
        final_score = compute_weighted_score(cos_sim[i], emb_sim[i], jacc, flav, rating, coverage)
        scores.append((i, final_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    columns = ["name","cuisine","ingredients","difficulty","rating"]
    if "flavor_profile" in df.columns:
        columns.append("flavor_profile")
    if "diet_type" in df.columns:
        columns.append("diet_type")
    for flag in ["is_vegetarian", "is_vegan", "is_dairy_free", "is_gluten_free"]:
        if flag in df.columns:
            columns.append(flag)
    
    result = df.loc[[i for i, _ in scores], columns].copy()
    result["score"] = [s for _, s in scores]
    
    # Only generate steps if explicitly requested
    if generate_steps:
        print("\n🔄 Generating recipes for all recommendations...")
        result["steps"] = result.apply(lambda row: generate_recipe_with_gemini(row["name"], row["ingredients"]), axis=1)
    else:
        result["steps"] = "Call get_recipe_steps() to generate"
    
    return result


def recommend_by_ingredients(user_ingredients, user_flavor, df, tfidf, tfidf_matrix, model, embeddings,
                             exclude_ingredients="", dietary_filters=None, top_n=5, generate_steps=False):
    """
    Recommend dishes based on ingredients. Set generate_steps=True to generate recipes immediately.
    """
    lemmatizer = WordNetLemmatizer()

    user_ingredients_str = str(user_ingredients) if user_ingredients else ""
    # Use lemmatized set of user ingredients for accurate comparison
    user_ingredients_set = set([lemmatizer.lemmatize(simple_spell_correction(i.strip().lower())) for i in user_ingredients_str.split(",") if i.strip()])
    user_clean = " ".join(user_ingredients_set)

    user_ingredients_list = user_ingredients_str # Keep original string for later checks

    print(f"\n🔍 Starting recommendation search...")
    print(f"  User wants: {user_ingredients}")
    # ... (other print statements)

    working_df = df.copy()
    initial_count = len(working_df)
    
    # --- Filter 1: Check for Non-Veg Preference ---
    if check_user_for_non_veg_preference(user_ingredients_str):
        print("\nApplying Non-Veg Preference filter: Filtering out vegetarian dishes.")
        before_non_veg = len(working_df)
        # Keep only dishes that are NOT marked as is_vegetarian
        working_df = working_df[working_df['is_vegetarian'] == False]
        after_non_veg = len(working_df)
        print(f"  After Non-Veg Preference: {after_non_veg} dishes remaining (removed {before_non_veg - after_non_veg} vegetarian dishes).")
        
        if len(working_df) == 0:
            print("❌ No non-vegetarian dishes found that match base criteria.")
            return pd.DataFrame()
    # --------------------------------------------------

    if dietary_filters:
        print(f"\nApplying dietary filters...")
        working_df = apply_dietary_filters(working_df, dietary_filters)
        if len(working_df) == 0:
            print("❌ No dishes match the dietary requirements")
            return pd.DataFrame()

    # --- FLAVOR FILTER LOGIC ---
    if user_flavor and user_flavor in flavor_dict:
        print(f"\nApplying preferred flavor filter: {user_flavor}")
        flavor_key = user_flavor
        
        # Filter for dishes where the preferred flavor is present (count > 0)
        before_flavor = len(working_df)
        working_df = working_df[working_df['flavor_profile'].apply(lambda x: x.get(flavor_key, 0) > 0)]
        after_flavor = len(working_df)
        print(f"  After flavor filter: {after_flavor} dishes remaining (removed {before_flavor - after_flavor})")
        
        if len(working_df) == 0:
            print(f"❌ No dishes found matching the flavor '{user_flavor}'")
            return pd.DataFrame()
    # -----------------------------

    exclude_str = str(exclude_ingredients) if exclude_ingredients else ""
    if exclude_str and exclude_str.strip().lower() != "none":
        print(f"\nApplying exclusion filter for: {exclude_str}")
        before_exclusion = len(working_df)
        working_df = working_df[~working_df["ingredients"].apply(lambda x: contains_excluded_ingredient(x, exclude_str))]
        after_exclusion = len(working_df)
        removed = before_exclusion - after_exclusion
        print(f"  After exclusion: {after_exclusion} dishes (removed {removed} dishes with '{exclude_str}')")
        if len(working_df) == 0:
            print(f"❌ No dishes found without '{exclude_str}'")
            return pd.DataFrame()

    before_protein = len(working_df)
    # Check_required_ingredients_present logic needs the original string list, not the cleaned one
    working_df = working_df[working_df["ingredients"].apply(lambda x: check_required_ingredients_present(x, user_ingredients_list))]
    after_protein = len(working_df)
    print(f"\nAfter protein matching: {after_protein} dishes (removed {before_protein - after_protein})")
    if len(working_df) == 0:
        print("❌ No dishes found with the ingredients you have")
        return pd.DataFrame()

    user_tfidf = tfidf.transform([user_clean])
    
    # Filter the matrices to match the working_df indices
    working_indices = working_df.index.tolist()
    
    full_cos_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    cos_sim = full_cos_sim[working_indices]
    
    user_emb = model.encode([user_clean], convert_to_tensor=True)
    full_emb_sim = util.cos_sim(user_emb, embeddings)[0].cpu().numpy()
    emb_sim = full_emb_sim[working_indices]

    profile = defaultdict(int)
    for flavor, keywords in flavor_dict.items():
        for k in keywords:
            if k in user_clean.split():
                profile[flavor] += 1
    
    if user_flavor and user_flavor in profile:
        profile[user_flavor] += 2

    scores = []
    
    # Iterate over the filtered working_df's index/data
    for idx_new, i in enumerate(working_df.index):
        dish_ingredients_str = df.loc[i, "ingredients"] # Use the raw ingredient string
        
        # Jaccard and Flavor Match logic remains the same (using clean/profile data)
        dish_ingredients_set = set(df.loc[i, "clean_ingredients"].split())
        jacc = jaccard_similarity(user_ingredients_set, dish_ingredients_set)
        flav = flavor_match(profile, df.loc[i, "flavor_profile"])
        rating = df.loc[i, "rating"] if "rating" in df.columns else 4.0
        
        # 🌟 NEW: Use the WEIGHTED COVERAGE function
        coverage = compute_ingredient_coverage(user_ingredients_set, dish_ingredients_str)
        
        final_score = compute_weighted_score(cos_sim[idx_new], emb_sim[idx_new], jacc, flav, rating, coverage)
        scores.append((i, final_score)) # i is the original DF index

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    columns = ["name","cuisine","ingredients","difficulty","rating"]
    if "flavor_profile" in df.columns:
        columns.append("flavor_profile")
    if "diet_type" in df.columns:
        columns.append("diet_type")
    for flag in ["is_vegetarian", "is_vegan", "is_dairy_free", "is_gluten_free"]:
        if flag in df.columns:
            columns.append(flag)
    
    result = df.loc[[i for i, _ in scores], columns].copy()
    result["score"] = [s for _, s in scores]
    
    # Only generate steps if explicitly requested
    if generate_steps:
        print("\n🔄 Generating recipes for all recommendations...")
        result["steps"] = result.apply(lambda row: generate_recipe_with_gemini(row["name"], row["ingredients"]), axis=1)
    else:
        result["steps"] = "Call get_recipe_steps() to generate"

    print(f"✅ Returning {len(result)} recommendations\n")
    return result


def build_system(file_path):
    print("\n🚀 Initializing Recipe Recommendation System with Gemini AI...")
    df = load_data(file_path)
    df = clean_and_lemmatize(df)
    df = compute_flavor_profile(df)
    df = classify_dietary_restrictions(df)
    
    # Verify Gemini API
    api_working = verify_gemini_api()
    if api_working:
        print("✅ Gemini AI configured for recipe generation")
    else:
        print("⚠️ Gemini AI unavailable - using template recipes")
    
    tfidf, tfidf_matrix = build_tfidf(df)
    model, embeddings = build_embeddings(df)
    return df, (tfidf, tfidf_matrix, model, embeddings)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🍽️  Recipe Recommender with Gemini AI")
    print("="*80)

    # Build the system
    # Assuming 'indian_dishes_modified.csv' is the correct dataset name used by Flask
    processed_df, (tfidf, tfidf_matrix, model, embeddings) = build_system("indian_dishes_dataset.csv") 

    # --- Test logic removed for brevity ---