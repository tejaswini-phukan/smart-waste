import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Set page config
st.set_page_config(
    page_title="♻️ Smart Waste Segregator Pro",
    page_icon="🗑️",
    layout="centered"
)

# Dataset
data = {
    'item': [
        # Biodegradable (20 items)
        'apple core', 'banana peel', 'egg shells', 'vegetable peels', 'wood chips',
        'tea leaves', 'coffee grounds', 'fruit scraps', 'bread crumbs', 'rice grains',
        'pasta leftovers', 'grass clippings', 'tree leaves', 'flower petals', 'hay',
        'straw', 'corn cobs', 'nut shells', 'potato peels', 'onion skins',
        
        # Recyclable (20 items)
        'plastic bottle', 'glass jar', 'newspaper', 'cardboard', 'aluminum can',
        'tetra pack', 'metal scraps', 'paper', 'magazines', 'books',
        'envelopes', 'junk mail', 'cereal boxes', 'shoe boxes', 'egg cartons',
        'milk cartons', 'juice boxes', 'soda cans', 'beer bottles', 'wine bottles',
        
        # Non-Biodegradable (20 items)
        'plastic bag', 'ceramic plate', 'cloth', 'rubber', 'styrofoam',
        'bubble wrap', 'packing peanuts', 'chip bags', 'candy wrappers', 'plastic wrap',
        'cling film', 'disposable diapers', 'sanitary pads', 'tampons', 'cotton swabs',
        'dental floss', 'bandages', 'medical waste', 'cigarette butts', 'chewing gum',
        
        # E-Waste (20 items)
        'battery', 'mobile phone', 'computer parts', 'laptop', 'tablet',
        'keyboard', 'mouse', 'monitor', 'television', 'printer',
        'scanner', 'router', 'modem', 'hard drive', 'flash drive',
        'cables', 'chargers', 'headphones', 'speakers', 'microwave',
        
        # Hazardous (20 items)
        'chemicals', 'paint', 'pesticides', 'herbicides', 'fertilizers',
        'motor oil', 'antifreeze', 'car battery', 'thermometer', 'thermostat',
        'fluorescent bulbs', 'CFL bulbs', 'mercury devices', 'asbestos', 'propane tanks',
        'fire extinguishers', 'nail polish', 'nail polish remover', 'aerosol cans', 'bleach'
    ],
    'category': [
        # Biodegradable (20)
        *['biodegradable']*20,
        
        # Recyclable (20)
        *['recyclable']*20,
        
        # Non-Biodegradable (20)
        *['non-biodegradable']*20,
        
        # E-Waste (20)
        *['e-waste']*20,
        
        # Hazardous (20)
        *['hazardous']*20
    ]
}

# Create and train model
@st.cache_resource
def get_model():
    df = pd.DataFrame(data)
    
    # Feature engineering
    df['item_length'] = df['item'].apply(len)
    df['has_plastic'] = df['item'].str.contains('plastic|bag|wrap|film').astype(int)
    df['has_metal'] = df['item'].str.contains('metal|aluminum|can|scraps|battery').astype(int)
    df['has_organic'] = df['item'].str.contains('apple|banana|egg|vegetable|wood|peel|leaves|fruit|grass|flower').astype(int)
    df['has_electronic'] = df['item'].str.contains('battery|phone|computer|laptop|tablet|keyboard|mouse|monitor|printer').astype(int)
    df['has_paper'] = df['item'].str.contains('paper|cardboard|newspaper|magazine|book|envelope').astype(int)
    df['has_chemical'] = df['item'].str.contains('chemical|paint|pesticide|herbicide|fertilizer|oil|mercury|asbestos').astype(int)
    
    # Encode target
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    
    # Features and target
    features = ['item_length', 'has_plastic', 'has_metal', 'has_organic', 
               'has_electronic', 'has_paper', 'has_chemical']
    X = df[features]
    y = df['category_encoded']
    
    # Train model
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    return model, le

model, le = get_model()

# Prediction function
def predict_waste(item):
    # Create features for the input
    item_length = len(item)
    has_plastic = 1 if 'plastic' in item.lower() or 'bag' in item.lower() or 'wrap' in item.lower() else 0
    has_metal = 1 if any(word in item.lower() for word in ['metal', 'aluminum', 'can', 'scraps', 'battery']) else 0
    has_organic = 1 if any(word in item.lower() for word in ['apple', 'banana', 'egg', 'vegetable', 'wood', 'peel', 'leaves', 'fruit']) else 0
    has_electronic = 1 if any(word in item.lower() for word in ['battery', 'phone', 'computer', 'laptop', 'tablet', 'electronic']) else 0
    has_paper = 1 if any(word in item.lower() for word in ['paper', 'cardboard', 'newspaper', 'magazine', 'book']) else 0
    has_chemical = 1 if any(word in item.lower() for word in ['chemical', 'paint', 'pesticide', 'herbicide', 'toxic']) else 0
    
    features = [[item_length, has_plastic, has_metal, has_organic, 
                has_electronic, has_paper, has_chemical]]
    
    prediction = model.predict(features)
    return le.inverse_transform(prediction)[0]

# Display function for results
def display_result(item, category):
    st.subheader("Classification Result")
    
    if category == 'biodegradable':
        st.success(f"**{item.title()}** is **Biodegradable** �")
        st.markdown("""
        - Can be composted
        - Includes food waste, yard trimmings, paper products
        - Breaks down naturally
        """)
        
    elif category == 'recyclable':
        st.info(f"**{item.title()}** is **Recyclable** ♻️")
        st.markdown("""
        - Place in recycling bin
        - Includes paper, glass, metal, certain plastics
        - Clean before recycling
        """)
        
    elif category == 'non-biodegradable':
        st.warning(f"**{item.title()}** is **Non-Biodegradable** 🚫")
        st.markdown("""
        - Should go in general waste
        - Will not decompose naturally
        - Consider reducing use of these items
        """)
        
    elif category == 'e-waste':
        st.error(f"**{item.title()}** is **E-Waste** 🔌")
        st.markdown("""
        - Requires special disposal
        - Take to e-waste collection points
        - Contains hazardous materials
        """)
        
    elif category == 'hazardous':
        st.error(f"**{item.title()}** is **Hazardous Waste** ☣️")
        st.markdown("""
        - Requires special handling
        - Do not dispose with regular trash
        - Contact local hazardous waste facility
        """)

# UI Components
st.title("♻️ Smart Waste Segregation Advisor")
st.markdown("Choose how you want to classify your waste:")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📝 Text Input", "📷 Image Upload"])

with tab1:
    # Text input functionality
    st.subheader("Enter waste item as text")
    with st.form("waste_form"):
        item = st.text_input("Describe the waste item (e.g., banana peel, plastic bottle):", 
                            placeholder="What do you want to dispose?")
        submitted = st.form_submit_button("Classify Waste")

    # Display text results
    if submitted:
        if item.strip() == "":
            st.warning("Please enter an item to classify")
        else:
            with st.spinner("Analyzing your waste item..."):
                category = predict_waste(item)
                display_result(item, category)

with tab2:
    # Image upload functionality
    st.subheader("Upload an image of waste")
    st.info("For best results, name your file with keywords (e.g., 'banana_peel.jpg', 'plastic_bottle.jpg')")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        
        # Simple filename-based recognition
        filename = uploaded_file.name.lower()
        
        # Check for keywords in filename
        waste_type = None
        
        # Biodegradable keywords
        if any(word in filename for word in ['apple', 'banana', 'peel', 'vegetable', 'fruit', 
                                           'egg', 'wood', 'leaves', 'grass', 'flower']):
            waste_type = "biodegradable"
            
        # Recyclable keywords
        elif any(word in filename for word in ['bottle', 'can', 'jar', 'paper', 'cardboard',
                                             'metal', 'aluminum', 'glass', 'newspaper']):
            waste_type = "recyclable"
            
        # Non-biodegradable keywords
        elif any(word in filename for word in ['plastic', 'bag', 'wrapper', 'styrofoam',
                                             'ceramic', 'rubber', 'cloth']):
            waste_type = "non-biodegradable"
            
        # E-waste keywords
        elif any(word in filename for word in ['battery', 'phone', 'laptop', 'electronic',
                                             'computer', 'tv', 'printer']):
            waste_type = "e-waste"
            
        # Hazardous keywords
        elif any(word in filename for word in ['chemical', 'paint', 'oil', 'thermometer',
                                             'mercury', 'asbestos', 'pesticide']):
            waste_type = "hazardous"
        
        if waste_type:
            st.success(f"From filename analysis: This appears to be {waste_type} waste")
            display_result(filename.replace('.jpg','').replace('_',' '), waste_type)
        else:
            st.warning("Could not determine waste type from filename. Please try text input or rename your file with keywords.")

# Add some examples
st.markdown("### Try these examples:")
cols = st.columns(5)
examples = ["banana peel", "plastic bottle", "newspaper", "battery", "paint"]
for col, example in zip(cols, examples):
    if col.button(example):
        category = predict_waste(example)
        st.info(f"**{example.title()}** is **{category.title()}**")

# Footer
st.markdown("---")
st.markdown("♻️ Proper waste segregation helps protect the environment and enables effective recycling.")
