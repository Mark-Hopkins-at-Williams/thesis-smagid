import React, { useState } from "react";
import "./App.css";
import googlefonts from "./assets/googlefonts.json"
import GoogleFontLoader from 'react-google-font'

const GlyphButton = ({ rotation, onClick, label, font_name }) => {
  return (
    <>
      <GoogleFontLoader fonts={[{font: font_name}]}/>
      <button className={`circle-button ${rotation}`} style={{ fontFamily: font_name }} onClick={onClick}>
        {label}
      </button>
    </>
  );
};

const CenterGlyph = ({ label, font_name }) => {
  console.log(font_name)
  return (
    <>
      <GoogleFontLoader fonts={[{font: font_name}]}/>
      <p className="center-glyph" style={{ fontFamily: font_name }}>
        {label}
      </p>
    </>
  )
}

const getRandomFont = () => {
  const randomIndex = Math.floor(Math.random() * googlefonts.items.length);
  const randomFont = googlefonts.items[randomIndex].family;
  return randomFont
};

const App = () => {
  // font state for all buttons
  const [fonts, setFonts] = useState(Array(6).fill().map(() => getRandomFont()));
  const [centerFont, setCenterFont] = useState(getRandomFont())

  const handleClick = (buttonIndex) => {
    const newFonts = Array(6).fill().map(() => getRandomFont())
    setFonts(newFonts);
    const newCenterFont = getRandomFont()
    setCenterFont(newCenterFont)
  };

  let letter = "A"
  
  return (
    <div className="parent">

      <h1 className = "title">Google Fontspace Selector</h1>

      <div className="circle-container">

        <CenterGlyph label={letter} font_name={centerFont} />

        {fonts.map((font, index) => (
          <GlyphButton
            key={index}
            rotation={"deg" + (index * 60)}
            onClick={() => handleClick(index)}
            label={letter}
            font_name={font}
          />
        ))}

      </div>
    </div>
  );
}

export default App