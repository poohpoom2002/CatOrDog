import React, { useState, useRef } from "react";
import './App.css';
function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictionR, setPredictionR] = useState("");
  const [predictionE, setPredictionE] = useState("");
  const [selectedImageURL, setSelectedImageURL] = useState("");
  const appRef = useRef();

  const handleFileChange = (e) => {
    setPredictionR("");
    setPredictionE("");
    const file = e.target.files[0];
    
    try {
      setSelectedImageURL(URL.createObjectURL(file));
      setSelectedFile(file);
    } catch (error) {
      console.error("Error creating object URL:", error);
    }
  };

  const handleUpload = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append("image", selectedFile);

      try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        setPredictionR(data.restNet);
        setPredictionE(data.efficientNet);
      } catch (error) {
        console.error("Error:", error);
      }
    }
  };

  return (
    <div className="App" ref={appRef}>
      <img src="https://i.ytimg.com/vi/IiilA0dsciY/maxresdefault.jpg" alt="background" className="imageBg"/>
      <h1>Image Classification</h1>
      <input
        className=""
        type="file"
        accept=".jpg, .jpeg, .png"
        onChange={handleFileChange}
        onClick = {()=>setSelectedFile(null)}
      />
      <button className={`button-17 ${(selectedFile && selectedFile != '') ? 'okay':''}`} onClick={handleUpload}>Let's Predict</button>
      <div className="predictionContainer">
        {predictionR && (
          <div className="predictionBox">
            <h2>Predicted Class RestNet50: <span style={{'fontSize':'80%', 'fontWeight': "normal"}}>{predictionR}</span></h2>
            <img src={selectedImageURL} alt="Selected Image" />
          </div>
        )}
        {predictionE && (
          <div className="predictionBox">
            <h2>Predicted Class from EfficientNetB5 : <span style={{'fontSize':'80%', 'fontWeight': "normal"}}>{predictionE}</span></h2>
            <img src={selectedImageURL} alt="Selected Image" />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
