import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import googlefonts from "./assets/googlefonts.json"
import GoogleFontLoader from 'react-google-font'
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import Slider from '@mui/material/Slider';
import Switch from '@mui/material/Switch';
import { styled } from '@mui/material/styles';
import ScatterPlot from './Scatterplot'
import { createTheme, ThemeProvider } from '@mui/material/styles';

const font1color = "rgb(255, 222, 248)"
const font2color = "rgb(250, 215, 171)"
const font3color = "rgb(255, 230, 176)"
const font4color = "rgb(214, 237, 214)"
const font5color = "rgb(202, 227, 255)"
const font6color = "rgb(221, 218, 255)"

const theme = createTheme({
  typography: {
    button: {
      textTransform: 'none',
      fontSize: 16
    }
  },
  palette: {
    grey: {
      main: 'rgb(113, 113, 113)',
      light: 'rgb(81, 81, 81)',
      dark: 'rgb(81, 81, 81)',
      contrastText: 'rgb(255, 255, 255)',
    },
  },
});

const iOSBoxShadow =
  '0 3px 1px rgba(0,0,0,0.1),0 4px 8px rgba(0,0,0,0.13),0 0 0 1px rgba(0,0,0,0.02)';

const IOSSlider = styled(Slider)(({ theme }) => ({
  color: '#828282',
  height: 250,
  padding: '0 0',
  '& .MuiSlider-thumb': {
    height: 20,
    width: 20,
    backgroundColor: '#fff',
    boxShadow: '0 0 2px 0px rgba(0, 0, 0, 0.1)',
    '&:focus, &:hover, &.Mui-active': {
      boxShadow: '0px 0px 3px 1px rgba(0, 0, 0, 0.1)',
      // Reset on touch devices, it doesn't add specificity
      '@media (hover: none)': {
        boxShadow: iOSBoxShadow,
      },
    },
    '&:before': {
      boxShadow:
        '0px 0px 1px 0px rgba(0,0,0,0.2), 0px 0px 0px 0px rgba(0,0,0,0.14), 0px 0px 1px 0px rgba(0,0,0,0.12)',
    },
  },
  '& .MuiSlider-valueLabel': {
    fontSize: 16,
    fontWeight: 'normal',
    top: -6,
    backgroundColor: 'unset',
    color: theme.palette.text.primary,
    '&::before': {
      display: 'none',
    },
    '& *': {
      background: 'transparent',
      color: '#000',
      ...theme.applyStyles('dark', {
        color: '#fff',
      }),
    },
  },
  '& .MuiSlider-track': {
    border: 'none',
    height: 5,
  },
  '& .MuiSlider-rail': {
    opacity: 0.5,
    boxShadow: 'inset 0px 0px 4px -2px #000',
    backgroundColor: '#d0d0d0',
  },
  ...theme.applyStyles('dark', {
    color: '#0a84ff',
  }),
}));


const GlyphButton = ({ rotation, onClick, label, fontName, setHoverFont, fonts }) => {
  let tooltipPlacement = 'left'

  if (rotation === 0 || rotation === 60 || rotation === 300) {
    tooltipPlacement = 'right'
  }

  let color = "#e0e0e0"

  if (fontName === fonts[0]) color = font1color;
  if (fontName === fonts[1]) color = font2color;
  if (fontName === fonts[2]) color = font3color;
  if (fontName === fonts[3]) color = font4color;
  if (fontName === fonts[4]) color = font5color;
  if (fontName === fonts[5]) color = font6color;

  return (
    <>
      <GoogleFontLoader fonts={[{font: fontName}]}/>
          <Tooltip
            title={<Typography sx={{ fontSize: '1rem' }}>{fontName}</Typography>}
            placement={tooltipPlacement}
          >
            <button
              className={`circle-button deg${rotation}`}
              style={{ fontFamily: fontName, backgroundColor: color}}
              onClick={onClick}
              onMouseEnter={() => setHoverFont(fontName)}
              onMouseLeave={() => setHoverFont("")}
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

const CenterGlyph = ({ label, fontName, onInput, setHoverFont }) => {
  return (
    <>
      <GoogleFontLoader fonts={[{font: fontName}]}/>
      <p
        className="center-glyph"
        style={{ fontFamily: fontName, backgroundColor: 'rgb(255, 171, 176)'}}
        contentEditable
        suppressContentEditableWarning
        onInput={onInput}
        onMouseEnter={() => setHoverFont(fontName)}
        onMouseLeave={() => setHoverFont("")}
      >
        {label}
      </p>
    </>
  )
}

const FontTitle = ({ fontName }) => {
  const formatFontName = (name) => name.replace(/\s+/g, '+');
  const fontUrl = `https://fonts.google.com/specimen/${formatFontName(fontName)}`;
  return (
    <a
      className="fontName"
      style={{ fontFamily: fontName }}
      href={fontUrl}
    >
      {fontName}
    </a>
  )
}

const getRandomFont = () => {
  const randomIndex = Math.floor(Math.random() * googlefonts.items.length)
  const randomFont = googlefonts.items[randomIndex].family
  return randomFont
}

const historyStack = []

const App = () => {
  const [magnitude, setMagnitude] = useState(0)
  const [data, setData] = useState(null)
  const [allFonts, setAllFonts] = useState([])
  const [fonts, setFonts] = useState(Array(6))
  const [centerFont, setCenterFont] = useState("")
  const [char, setChar] = useState("A")
  const [hoverFont, setHoverFont] = useState("")

  useEffect(() => { // this runs only once
    fetch("http://appa.cs.williams.edu:18812/allfonts")
      .then((response) => response.json())
      .then((jsonData) => {
        setAllFonts(jsonData)
      })
    .catch((error) => {
      console.error("Error fetching data:", error)
    })
  }, [])

  const fetchData = async (mag, font) => {
    let queryURL = `http://appa.cs.williams.edu:18812/getfont?mag=${mag}`
    if (font !== "") {
      queryURL += `&center=${encodeURIComponent(font)}`
    }
    fetch(queryURL)
      .then((response) => response.json())
      .then((data) => {
        setCenterFont(data.centerFont)
        setFonts(data.selectedFonts)
      })
      .catch((error) => console.error("Error fetching data:", error))
  }

  useEffect(() => {
    fetchData(magnitude, centerFont)
  }, [historyStack, magnitude, centerFont])

  useEffect(() => {
    setMagnitude(0)
  }, [centerFont])

  const shuffle = () => {
    historyStack.push([...fonts, centerFont, magnitude]) // save to history
    setMagnitude(0)
    fetchData(magnitude, "")
  }

  const back = () => {
    if (historyStack.length > 0) {
      const lastFonts = historyStack.pop()
      const magnitude = lastFonts.pop()
      const lastCenterFont = lastFonts.pop()
      console.log('backing up')
      setMagnitude(magnitude)
      setCenterFont(lastCenterFont)
      setFonts(lastFonts)
    } else {
      console.log('history is empty')
    }
  }

  const handleGlyphButtonClick = (buttonIndex) => {
    historyStack.push([...fonts, centerFont, magnitude]) // save to history
    setCenterFont(fonts[buttonIndex])
  }

  const handleScatterClick = (chosenFont) => {
    historyStack.push([...fonts, centerFont, magnitude]) // save to history
    setCenterFont(chosenFont)
  }

  const handleSlider = (event, newValue) => {
    setMagnitude(newValue)
  }

  const resetSlider = () => {
    setMagnitude(0)
  }

  const handleInput = (e) => {
    let text = e.target.innerText
    if (text.length > 1) {
      text = text.charAt(text.length - 1)
    }
    setChar(text)
    e.target.innerText = text
  }
  
  return (
    <ThemeProvider theme={theme}>
      <div className="parent">

        <GoogleFontLoader fonts={allFonts}/>

        <h1 className = "title">Typeface Space Selector (Google Fonts)</h1>

        <div className="horizontal">

          <div className="left-box">
            <ScatterPlot
              fonts={fonts}
              centerFont={centerFont}
              handleScatterClick={handleScatterClick}
              chosenCharacter={char}
              hoverFont={hoverFont}
              magnitude={magnitude}
            />
          </div>

          <div className="center-box">
            <div className="sliderBox">
              <IOSSlider 
                className="slider"
                aria-label="Slider"
                value={magnitude}
                track={false}
                onChange={handleSlider}
                orientation="vertical"
                min={-8}
                max={8}
                step={0.1}
                valueLabelDisplay="off"
              />
              <div className="sliderLabelBox">
                <Tooltip title={<Typography sx={{ fontSize: '0.8rem' }}>Reset Magnitude</Typography>} placement='bottom'>
                  <p onClick={resetSlider} style={{cursor: 'pointer'}}>{magnitude}</p>
                </Tooltip>
              </div>
            </div>
          </div>

          <div className="right-box">

            <div className="circle-container">

              <CenterGlyph label={char} fontName={centerFont} onInput={handleInput} setHoverFont={setHoverFont}/>

              {fonts.map((font, index) => (
                <GlyphButton
                  key={index}
                  rotation={index * 60}
                  onClick={() => handleGlyphButtonClick(index)}
                  label={char}
                  fontName={font}
                  setHoverFont={setHoverFont}
                  fonts={fonts}
                />
              ))}

            </div>

            <FontTitle fontName={centerFont}></FontTitle>

            <div className="utility-button-container">
              <BackButton onClick={back}></BackButton>
              <ShuffleButton onClick={shuffle}></ShuffleButton>
            </div>

          </div>

        </div>

      </div>
    </ThemeProvider>
  )
}

export default App