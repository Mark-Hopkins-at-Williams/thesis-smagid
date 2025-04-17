import React, { useEffect, useState, useRef } from "react";
import { Scatter } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, Title, Tooltip, Legend } from "chart.js";
import GoogleFontLoader from 'react-google-font';
import useFontFaceObserver from 'use-font-face-observer';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import zoomPlugin from 'chartjs-plugin-zoom';
import Switch from '@mui/material/Switch';
import Button from '@mui/material/Button';

ChartJS.register(CategoryScale, LinearScale, PointElement, Title, Tooltip, Legend, ChartDataLabels, zoomPlugin);

const lightblue = "rgb(159, 221, 221)"
const red = "rgb(235, 64, 52)"
const yellow = "rgb(237, 197, 64)"
const lightred = "rgb(239, 143, 136)"
const lightyellow = "rgb(240, 205, 134)"

const font1color = "rgb(255, 134, 229)"
const font2color = "rgb(255, 160, 37)"
const font3color = "rgb(255, 209, 69)"
const font4color = "rgb(125, 212, 131)"
const font5color = "rgb(145, 198, 255)"
const font6color = "rgb(147, 135, 255)"

const ScatterPlot = ({ fonts, centerFont, handleScatterClick, chosenCharacter, hoverFont, magnitude }) => {
    const [data, setData] = useState([])
    const [loading, setLoading] = useState(true);
    const zoomBounds = useRef(null);
    const chartRef = useRef(null);
    const [displayChars, setDisplayChars] = useState(false);
    
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

    const scatterData = {
      datasets: [
          {
              label: "Dataset",
              data: data.map((item) => ({
                  x: item.x,
                  y: item.y,
                  font: item.font,
              })),
              pointRadius: (context) => {
                let size = 4
                const font = context.dataset.data[context.dataIndex].font;
                if (font === hoverFont) {
                  size = 7
                }
                if (chartRef.current) {
                  const zoomScale = chartRef.current.getZoomLevel(chartRef)
                  size = size * Math.sqrt(zoomScale)
                }
                return size
              },
              pointHoverRadius: () => {
                let size = 7
                if (chartRef.current) {
                  const zoomScale = chartRef.current.getZoomLevel(chartRef)
                  return size * Math.sqrt(zoomScale)
                }
                return size
              },
              pointBackgroundColor: displayChars 
              ? "rgba(0, 0, 0, 0)" 
              : (context) => {
                  const font = context.dataset.data[context.dataIndex].font;
                  if (font === centerFont) return red;
                  if (font === fonts[0]) return font1color;
                  if (font === fonts[1]) return font2color;
                  if (font === fonts[2]) return font3color;
                  if (font === fonts[3]) return font4color;
                  if (font === fonts[4]) return font5color;
                  if (font === fonts[5]) return font6color;
                  return "rgb(176, 176, 176, 0.3)";
                },
              pointBorderColor: displayChars 
              ? "rgba(0, 0, 0, 0)" 
              : (context) => {
                  const font = context.dataset.data[context.dataIndex].font;
                  if (font === centerFont) return red;
                  if (font === fonts[0]) return font1color;
                  if (font === fonts[1]) return font2color;
                  if (font === fonts[2]) return font3color;
                  if (font === fonts[3]) return font4color;
                  if (font === fonts[4]) return font5color;
                  if (font === fonts[5]) return font6color;
                  return "rgb(120, 120, 120, 0)";
                },
              datalabels: {
                display: displayChars,
              },
          },
      ],
  };

    const options = {
      animation: false,
      responsive: true,
      layout: {
        padding: {
          top: 10,
          bottom: 10,
          left: 10,
          right: 10,
        },
      },
      onClick: function(event, chartElements) {
        if (this.tooltip._active.length > 0) {
          var font = this.tooltip.body[0].lines[0]
          handleScatterClick(font)
        }
      },
      plugins: {
        zoom: {
          pan: {
              enabled: true,
              mode: 'xy',
              onPan: (chart) => {
                zoomBounds.current = chart.chart.getZoomedScaleBounds(chart)
              }
          },
          zoom: {
              mode: 'xy',
              wheel: { enabled: true },
              pinch: { enabled: true },
              onZoom: (chart) => {
                // console.log('chart:',chart.chart)
                zoomBounds.current = chart.chart.getZoomedScaleBounds(chart)
              },
          },
        },
        legend: {
          display: false,
        },
        datalabels: {
          align: 'center',
          anchor: 'center',
          color: (context) => {
            var font = context.dataset.data[context.dataIndex].font;
            if (font === centerFont) return red;
            if (font === fonts[0]) return font1color;
            if (font === fonts[1]) return font2color;
            if (font === fonts[2]) return font3color;
            if (font === fonts[3]) return font4color;
            if (font === fonts[4]) return font5color;
            if (font === fonts[5]) return font6color;
            else return "black"
          },
          font: function(context) {
            var family = context.dataset.data[context.dataIndex].font
            const zoomScale = chartRef.current.getZoomLevel(chartRef)
            // THIS MAKES THINGS SLOW
            var size = 12
            if (family === hoverFont) {
              size = 24
            }
            size = Math.round(size * zoomScale ** .6)
            // var size = 12
            return {
              family: family,
              size: size,
              weight: 'bold',
            };
          },
          formatter: () => chosenCharacter,
        },
        tooltip: {
          mode: 'nearest',
          intersect: true,
          displayColors: false,
          backgroundColor: "rgb(247, 247, 247)",
          padding: 10,
          borderWidth: 3,
          borderColor: function (tooltipItem) {
            var font = tooltipItem.tooltipItems[0].raw.font
            if (font === centerFont) return red;
            if (font === fonts[0]) return font1color;
            if (font === fonts[1]) return font2color;
            if (font === fonts[2]) return font3color;
            if (font === fonts[3]) return font4color;
            if (font === fonts[4]) return font5color;
            if (font === fonts[5]) return font6color;
            else return "rgb(247, 247, 247)"
          },
          bodyColor: '#000000',
          bodyFont: {
            family: (tooltipItems) => {
              const font = tooltipItems.tooltipItems[0]?.raw?.font || "Roboto";
              return font;
            },
            size: 20,
          },
          callbacks: {
            label: function (tooltipItem) {
              const font = tooltipItem.raw.font
              return font
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

    useEffect(() => {
      if (chartRef && zoomBounds.current) {
        if (zoomBounds.current.x && zoomBounds.current.y) {
          console.log('setting bounds')
          chartRef.current.zoomScale('x', zoomBounds.current.x)
          chartRef.current.zoomScale('y', zoomBounds.current.y)
        }
      }
    }, [centerFont, chosenCharacter, displayChars, hoverFont, magnitude, data, fonts]);

    const resetZoom = () => {
      if (chartRef.current) {
          chartRef.current.resetZoom();
          zoomBounds.current = chartRef.current.getZoomedScaleBounds(chartRef)
      }
    }

    const handleSwitch = (event) => {
      setDisplayChars(event.target.checked);
    };

    if (loading) {
      return <div>Loading...</div>;
    }

    return (
      <div className="scatterplotBox">
        <Scatter ref={chartRef} data={scatterData} options={options} width={800} height={600} />
        <div className="scatterplotOptions">
            <p>View Characters:</p>
            <Switch
              checked={displayChars}
              onChange={handleSwitch}
              name="Display Characters"
              color="black"
            />
            <div className="scatterplotButton">
              <Button variant="contained" color="grey" onClick={() => {resetZoom()}}>
                Reset Zoom
              </Button>
            </div>
        </div>
      </div>
    )
}

export default ScatterPlot;