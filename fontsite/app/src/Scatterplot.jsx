import React, { useEffect, useState } from "react";
import { Scatter } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, Title, Tooltip, Legend } from "chart.js";
import GoogleFontLoader from 'react-google-font';
import useFontFaceObserver from 'use-font-face-observer';

ChartJS.register(CategoryScale, LinearScale, PointElement, Title, Tooltip, Legend);

const ScatterPlot = ({ fonts, centerFont, handleScatterClick }) => {
    const [data, setData] = useState([])
    const [loading, setLoading] = useState(true);
    const [hoverFont, sethoverFont] = useState("Roboto")
    
    // fetch data (only once)
    useEffect(() => {
      fetch("http://appa.cs.williams.edu:18812/gettsne")
        .then((response) => response.json())
        .then((jsonData) => {
          setData(jsonData)
          setLoading(false)
        })
      .catch((error) => {
        console.error("Error fetching data:", error)
        setLoading(false)
      })
    }, [])

    const useFont = (fontName) => {
      const [isFontLoaded, setIsFontLoaded] = useState(false);
    
      useEffect(() => {
        const checkFont = async () => {
          // Check if the font is already loaded
          if (document.fonts.check(`1em ${fontName}`)) {
            setIsFontLoaded(true);
            console.log('yay!')
          } else {
            // Wait for the font to load
            await document.fonts.load(`1em ${fontName}`);
            setIsFontLoaded(true);
          }
        };
    
        checkFont();
      }, [fontName]);
    
      return isFontLoaded;
    }

    // THIS IS SAYING FONT IS LOADED BUT TOOLTIP IS STILL NOT ACCESSING
    const isFontLoaded = useFont(hoverFont);

    const scatterData = {
      datasets: [
          {
              label: "Dataset",
              data: data.map((item) => ({
                  x: item.x,
                  y: item.y,
                  font: item.font,
              })),
              pointRadius: 4,
              pointBackgroundColor: function(context) {
                var font = context.raw.font
                if (font == centerFont) return "rgb(235, 64, 52)"
                if (fonts.includes(font)) return "rgb(237, 197, 64)"
                else return "rgba(75,192,192,0.2)"
              },
              pointBorderColor: function(context) {
                var font = context.raw.font
                if (font == centerFont) return "rgb(235, 64, 52)"
                if (fonts.includes(font)) return "rgb(237, 197, 64)"
                else return "rgba(75,192,192,0)"
              },
          },
      ],
  };

    const options = {
      animation: false,
      responsive: true,
      onClick: function(event, chartElements) {
        if (this.tooltip._active.length > 0) {
          handleScatterClick(hoverFont)
        }
      },
      plugins: {
        legend: {
          display: false,  // Hide the legend
        },
        tooltip: {
          mode: 'nearest',
          intersect: true,
          displayColors: false,
          backgroundColor: "rgb(247, 247, 247)",
          padding: 10,
          borderWidth: 4,
          borderColor: function (tooltipItem) {
            var font = tooltipItem.tooltipItems[0].raw.font
            if (font == centerFont) return "rgb(239, 143, 136)"
            if (fonts.includes(font)) return "rgb(240, 205, 134)"
            else return "rgb(247, 247, 247)"
          },
          bodyColor: '#000000',
          bodyFont: {
            family: hoverFont || 'Roboto',
            size: 20
          },
          callbacks: {
            label: function (tooltipItem) {
              const font = tooltipItem.raw.font
              sethoverFont(font)
              console.log(isFontLoaded)
              return isFontLoaded ? font : ''
            },
          },
        },
      },
      scales: {
        x: {
          display: false,
          grid: {
            display:false
          },
        },
        y: {
          display: false,
          grid: {
            display:false
          },
        },
      },
    }

    if (loading) {
      return <div>Loading...</div>;
    }

    console.log(`GoogleFontLoader is rendering with font: ${hoverFont}`);

    return (
      <div>
        <GoogleFontLoader fonts={[{font: hoverFont}]}/>
        <Scatter data={scatterData} options={options} width={800} height={600} />
      </div>
    )
}

export default ScatterPlot;