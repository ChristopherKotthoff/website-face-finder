import cv2
import insightface
import numpy as np

class Recognizer:
    def __init__(self):
        self.model = insightface.app.FaceAnalysis(name='buffalo_sc', root='.insightface', providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=0)

    def get_faces(self, img):
        return self.model.get(img)

    def cosine_similarity(self, embedding1, embedding2):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
def draw_bounding_box(img, face, similarity_score):
    bbox = face.bbox.astype(int)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    text = f"Sim: {similarity_score:.2f}"
    cv2.putText(img, text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, max(0.35,(bbox[2]-bbox[0])*0.01), (0, 255, 0), 1)
    
    
    
#Test
if __name__ == '__main__':
    recognizer = Recognizer()
    image_searched = cv2.imread('image1.jpg')
    image_strangers = cv2.imread('image2.jpg')
    face_searched = recognizer.get_faces(image_searched)[0]
    faces_strangers = recognizer.get_faces(image_strangers)
    
    for stranger_face in faces_strangers:
        similarity = recognizer.cosine_similarity(face_searched.embedding, stranger_face.embedding)
        draw_bounding_box(image_strangers, stranger_face, similarity)
    
    cv2.imwrite('output.jpg', image_strangers)
    print("Output saved as 'output.jpg'.")