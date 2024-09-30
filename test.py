import cv2
import insightface
import numpy as np

# Load the pre-trained model
model = insightface.app.FaceAnalysis(name='buffalo_sc', root='.insightface')
model.prepare(ctx_id=0)

# Load images
img1 = cv2.imread('face1.jpg')
img2 = cv2.imread('face2.jpg')

# Get face embeddings
faces1 = model.get(img1)
faces2 = model.get(img2)

# Check if faces were detected
if len(faces1) == 1 and len(faces2) > 0:
    embedding1 = faces1[0].embedding

    # Function to calculate cosine similarity
    def cosine_similarity(embedding1, embedding2):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        return similarity

    # Loop through each detected face in img2
    for index, face in enumerate(faces2):
        embedding2 = face.embedding
        similarity_score = cosine_similarity(embedding1, embedding2)
        print(f"Cosine similarity {index}: {similarity_score}")

        # Draw the bounding box around the detected face
        bbox = face.bbox.astype(int)  # Get bounding box coordinates
        cv2.rectangle(img2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # Put the similarity score next to the bounding box
        text = f"Sim: {similarity_score:.2f}"
        cv2.putText(img2, text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, max(0.35,(bbox[2]-bbox[0])*0.01), (0, 255, 0), 1)

    # Save the output image with bounding boxes and similarity scores
    cv2.imwrite('output.jpg', img2)
    print("Output saved as 'output.jpg'.")
else:
    print("Face not detected in one or both images.")
