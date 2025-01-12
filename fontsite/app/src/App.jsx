import React from "react";
import ReactFontLoader from 'react-font-loader'
import "./App.css";

const App = () => {
  const [count, setCount] = React.useState(0);

  const handleClick = () => {
    setCount(count + 1);
  };
  
  return (
    <div className="parent">
      <ReactFontLoader url="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100..900;1,100..900&display=swap" />

      <h1 className = "title">Google Fontspace Selector</h1>

      <div className="circle-container">

        <ReactFontLoader url="https://fonts.googleapis.com/css2?family=Rubik+Vinyl&display=swap" />

        <p className="center-glyph" style={{ fontFamily: 'Rubik Vinyl' }}>A</p>

        <button className="circle-button deg0" onClick={handleClick}>
          A
        </button>

        <button className="circle-button deg60" onClick={handleClick}>
          A
        </button>

        <button className="circle-button deg120" onClick={handleClick}>
          A
        </button>

        <button className="circle-button deg180" onClick={handleClick}>
          A
        </button>

        <button className="circle-button deg240" onClick={handleClick}>
          A
        </button>

        <button className="circle-button deg300" onClick={handleClick}>
          A
        </button>
      </div>
    </div>
  );
}

export default App