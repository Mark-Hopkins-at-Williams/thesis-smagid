import React, { useEffect, useState } from "react";
import { Scatter } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, Title, Tooltip, Legend } from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, Title, Tooltip, Legend);

const ScatterPlot = () => {
    const [data, setData] = useState([])
    const [loading, setLoading] = useState(true);
    const [chosenFont, setChosenFont] = useState("")
    
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
          backgroundColor: "rgba(75,192,192,1)",
          borderColor: "rgba(75,192,192,1)",
          borderWidth: 1,
        },
      ],
    }

    const options = {
      animation: false,
      responsive: true,
      onClick: function(event, chartElements) {
        if (this.tooltip._active.length > 0) {
          console.log(chosenFont)
        }
      },
      plugins: {
        legend: {
          display: false,  // Hide the legend
        },
        tooltip: {
          mode: 'nearest',
          intersect: true,
          callbacks: {
            label: function (tooltipItem) {
              const font = tooltipItem.raw.font
              setChosenFont(font)
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

    if (loading) {
      return <div>Loading...</div>;
    }

    return (
      <div>
        <Scatter data={scatterData} options={options} width={800} height={600} />
      </div>
    )
}

export default ScatterPlot;