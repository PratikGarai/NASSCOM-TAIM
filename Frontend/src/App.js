import React, { useRef, useEffect } from "react";
import mapboxgl from "mapbox-gl";

import "./App.css";

mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_ACCESS_TOKEN;

const App = () => {
  const mapContainerRef = useRef(null);
  // initialize map when component mounts
  useEffect(() => {
    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: "mapbox://styles/mapbox/streets-v11",
      center: [79.0193, 18.1124],
      zoom: 7
    });

    // add navigation control (zoom buttons)
    map.addControl(new mapboxgl.NavigationControl(), "bottom-right");

    map.on("load", () => {
      map.addSource("heatmap-points-data", {
        type: "geojson",
        data:'https://docs.mapbox.com/mapbox-gl-js/assets/earthquakes.geojson'
        //DATA GOES HERE
      });
      // now add the layer, and reference the data source above by name
      map.addLayer({
        id: "heatmap-points-layer",
        source: "heatmap-points-data",
        type:"heatmap",
      });
    });
    // clean up on unmount
    return () => map.remove();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return <div className="map-container" ref={mapContainerRef} />;
};

export default App;
