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

const ScatterPlot = ({ fonts, centerFont, handleScatterClick, chosenCharacter, hoverFont }) => {
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
              pointRadius: () => {
                if (chartRef.current) {
                  const zoomScale = chartRef.current.getZoomLevel(chartRef)
                  return 4 * Math.sqrt(zoomScale)
                }
                return 4
              },
              pointBackgroundColor: displayChars 
              ? "rgba(0, 0, 0, 0)" 
              : (context) => {
                  const font = context.dataset.data[context.dataIndex].font;
                  if (font === centerFont) return "rgb(235, 64, 52)";
                  if (font === hoverFont) return "rgb(117, 214, 214)";
                  if (fonts.includes(font)) return "rgb(237, 197, 64)";
                  return "rgb(176, 176, 176, 0.3)";
                },
              pointBorderColor: displayChars 
              ? "rgba(0, 0, 0, 0)" 
              : (context) => {
                  const font = context.dataset.data[context.dataIndex].font;
                  if (font === centerFont) return "rgb(235, 64, 52)";
                  if (font === hoverFont) return "rgb(117, 214, 214)";
                  if (fonts.includes(font)) return "rgb(237, 197, 64)";
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
            if (font == centerFont) return "rgb(235, 64, 52)"
            if (font === hoverFont) return "rgb(117, 214, 214)"
            if (fonts.includes(font)) return "rgb(237, 197, 64)"
            else return "black"
          },
          font: function(context) {
            var family = context.dataset.data[context.dataIndex].font
            const zoomScale = chartRef.current.getZoomLevel(chartRef)
            // var size = 12 * Math.sqrt(zoomScale) THIS MAKES THINGS SLOW
            var size = 12
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
            if (font == centerFont) return "rgb(239, 143, 136)"
            if (fonts.includes(font)) return "rgb(240, 205, 134)"
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
          chartRef.current.zoomScale('x', zoomBounds.current.x)
          chartRef.current.zoomScale('y', zoomBounds.current.y)
        }
      }
    }, [centerFont, chosenCharacter, displayChars, hoverFont]);

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
              <Button variant="contained" color="grey"   onClick={() => {resetZoom()}}>
                Reset Zoom
              </Button>
            </div>
        </div>
      </div>
    )
}

export default ScatterPlot;