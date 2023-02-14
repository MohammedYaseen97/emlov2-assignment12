import torch
import json
import torchvision.transforms as T
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# data transformations
transforms = T.Compose([
    # T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


def model_fn(model_dir):
    model = torch.jit.load(f"{model_dir}/model.scripted.pt")

    model.to(device).eval()

    return model

# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    data = json.loads(request_body)["inputs"]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    data = transforms(data)
    return data


# inference
def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    # res = predictions.cpu().numpy().tolist()
    
    ps = F.softmax(predictions, dim=-1).cpu()
    probs, classes = torch.topk(ps, 5, dim=1)
    probs = probs.tolist()
    classes = classes.tolist()
    
    output_list = []
    for i in range(len(predictions)):
        output_list.append({labels[class_]: prob for prob, class_ in zip(probs[i], classes[i])})

    # return json.dumps({labels[class_]: prob for prob, class_ in zip(probs, classes)})
    return json.dumps(output_list)
