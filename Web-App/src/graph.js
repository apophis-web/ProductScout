import React, { Component } from 'react';
import CanvasJSReact from './canvasjs.react';
var CanvasJS = CanvasJSReact.CanvasJS;
var CanvasJSChart = CanvasJSReact.CanvasJSChart;
 
class Graph extends Component {
	render() {
		const options = {
            interactivityEnabled: false,
            animationEnabled: true,
            animationDuration: 1500,
			exportEnabled: false,
			theme: "light1",
            backgroundColor: "#1D1D1F",
            axisY: {
                gridThickness: 0,
                tickThickness: 0,
                labelFormatter: function () {
                    return "";
                  }
              },
              axisX: {
                interval: 2,
                gridThickness: 0,
                tickThickness: 0,
                labelFormatter: function () {
                    return "";
                  }
              },
			data: [{
				type: "line",
                lineThickness: 3,
				toolTipContent: "Week {x}: {y}%",
                color: "white",
                dataPoints: Array.from({length: 200}, (_, i) => ({
                    x: i + 1,
                    y: Math.floor(Math.random() * 100)
                  }))
			}]
		}
		return (
            <div>
                <CanvasJSChart options = {options}/>
            </div>
		);
	}
}
       
export default Graph