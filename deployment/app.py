from fastai.vision.all import load_learner
import gradio as gr

elec_labels=(
    'Air Conditioner appliances',
    'Air Fryer appliances',
    'Blender appliances',
    'Clothes Dryer appliances',
    'Coffee Maker appliances',
    'Dish Washer appliances',
    'Electric Guitar appliances',
    'Electric toothbrush appliances',
    'Fan appliances',
    'Griller appliances',
    'Hair dryer appliances',
    'Heater appliances',
    'Induction cooktop appliances ',
    'Iron appliances',
    'Kettle appliances',
    'Microwave appliances',
    'Mixer appliances',
    'Refrigerator appliances',
    'Rice Cooker appliances',
    'Speaker appliances',
    'Toaster appliances',
    'Vacuum Cleaner appliances'
)

model=load_learner(f'models/elec-recognizer-v4.pkl')


def recognize_image(image):
  # image = input_image.resize((192, 192))
  pred,idx,probs=model.predict(image)
  return dict(zip(elec_labels,map(float,probs)))



image = gr.Image()
label = gr.Label()
examples=[
    'test_images/unknown_00.jpg',
    'test_images/unknown_01.jpg.webp',
    'test_images/unknown_02.webp',
    'test_images/unknown_03.jpeg',
    'test_images/unknown_04.jpg'
]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label,examples=examples)
iface.launch(inline=False)