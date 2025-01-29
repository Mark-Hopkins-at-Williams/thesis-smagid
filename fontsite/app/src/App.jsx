import React, { useState, useRef } from "react";
import "./App.css";
import googlefonts from "./assets/googlefonts.json"
import GoogleFontLoader from 'react-google-font'
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import Slider from '@mui/material/Slider';

const GlyphButton = ({ rotation, onClick, label, font_name }) => {
  let tooltipPlacement = 'left'

  if (rotation === 0 || rotation === 60 || rotation === 300) {
    tooltipPlacement = 'right'
  }

  return (
    <>
      <GoogleFontLoader fonts={[{font: font_name}]}/>
          <Tooltip
            title={<Typography sx={{ fontSize: '1rem' }}>{font_name}</Typography>}
            placement={tooltipPlacement}
          >
            <button
              className={`circle-button deg${rotation}`}
              style={{ fontFamily: font_name }}
              onClick={onClick}
            >
              {label}
            </button>
          </Tooltip>
    </>
  )
}

const BackButton = ( {onClick} ) => {
  return (
    <Tooltip
      title={<Typography sx={{ fontSize: '1rem' }}>Back</Typography>}
      placement='bottom'
    >
      <button
        className='utility-button'
        onClick={onClick}
      >
        <img src="src/assets/back.svg"/>
      </button>
    </Tooltip>
  )
}

const ShuffleButton = ( {onClick} ) => {
  return (
    <Tooltip
      title={<Typography sx={{ fontSize: '1rem' }}>Shuffle</Typography>}
      placement='bottom'
    >
      <button
        className='utility-button'
        onClick={onClick}
      >
        <img src="src/assets/shuffle.svg"/>
      </button>
    </Tooltip>
  )
}

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

const historyStack = []

const App = () => {
  // font state for all buttons
  const [fonts, setFonts] = useState(Array(6).fill().map(() => getRandomFont()));
  const [centerFont, setCenterFont] = useState(getRandomFont())
  const [sliderValue, setSliderValue] = useState(0)

  const shuffle = () => {
    historyStack.push([...fonts, centerFont]) // save fonts to history
    const newCenterFont = getRandomFont()
    setCenterFont(newCenterFont)
    const newFonts = Array(6).fill().map(() => getRandomFont())
    setFonts(newFonts)
  }

  const back = () => {
    if (historyStack.length > 0) {
      const lastFonts = historyStack.pop()
      const lastCenterFont = lastFonts.pop()
      console.log('backing up' + lastCenterFont)
      setCenterFont(lastCenterFont)
      setFonts(lastFonts)
    } else {
      console.log('history is empty')
    }
  }

  const handleClick = (buttonIndex) => {
    historyStack.push([...fonts, centerFont]) // save fonts to history
    const newCenterFont = fonts[buttonIndex]
    setCenterFont(newCenterFont)
    const newFonts = Array(6).fill().map(() => getRandomFont())
    setFonts(newFonts)
  }

  const handleSlider = (event, newValue) => {
    setSliderValue(newValue);
  };

  let letter = "A"
  
  return (
    <div className="parent">

      <h1 className = "title">Google Fontspace Selector</h1>

      <div className="horizontal">

        <div className="left-box"></div>

        <div className="center-box">

          <div className="circle-container">

            <CenterGlyph label={letter} font_name={centerFont} />

            {fonts.map((font, index) => (
              <GlyphButton
                key={index}
                rotation={index * 60}
                onClick={() => handleClick(index)}
                label={letter}
                font_name={font}
              />
            ))}

          </div>

          <p className="fontName">{centerFont}</p>

          <div className="utility-button-container">
            <BackButton onClick={back}></BackButton>
            <ShuffleButton onClick={shuffle}></ShuffleButton>
          </div>

        </div>

        <div className="right-box">
          <Tooltip
          title={<Typography sx={{ fontSize: '1rem' }}>Magnitude</Typography>}
          placement='bottom'
          >
            <Slider 
              className="slider"
              aria-label="Slider"
              value={sliderValue}
              onChange={handleSlider}
              min={-100}
              max={100}
              color="#828282"
            />
          </Tooltip>
        </div>

      </div>

    </div>
  )
}

export default App