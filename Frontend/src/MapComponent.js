import ReactDOM from "react-dom";
import React, { useRef, useEffect } from "react";
import mapboxgl from "mapbox-gl";

import fetchFakeData from "./api/fetchFakeData";
import Popup from "./components/Popup";
import "./App.css";

mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_ACCESS_TOKEN;

const MapComponent = () => {
  const mapContainerRef = useRef(null);
  const popUpRef = useRef(new mapboxgl.Popup({ offset: 15 }));

  // initialize map when component mounts
  useEffect(() => {
    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      // See style options here: https://docs.mapbox.com/api/maps/#styles
      style: "mapbox://styles/mapbox/streets-v11",
      center: [79.0193,18.1124],
      zoom: 7
    });

    // add navigation control (zoom buttons)
    map.addControl(new mapboxgl.NavigationControl(), "bottom-right");

    map.on("load", () => {
      // add the data source for new a feature collection with no features
      map.addSource("random-points-data", {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: []
        }
      });
      // now add the layer, and reference the data source above by name
      map.addLayer({
        id: "random-points-layer",
        source: "random-points-data",
        type: "symbol",
        layout: {
          // full list of icons here: https://labs.mapbox.com/maki-icons
          "icon-image": "bakery-15", // this will put little croissants on our map
          "icon-padding": 0,
          "icon-allow-overlap": true
        }
      });
      map.addLayer({
        id: "heatmap-layer",
        source: "random-points-data",
        type:"heatmap",
      });
    });

    map.on("moveend", async () => {
      const { lng, lat } = map.getCenter();
      const results = await fetchFakeData({ longitude: lng, latitude: lat });
      map.getSource("random-points-data").setData(results);
    });

    map.on("mouseenter", "random-points-layer", e => {
      if (e.features.length) {
        map.getCanvas().style.cursor = "pointer";
      }
    });

    map.on("mouseleave", "random-points-layer", () => {
      map.getCanvas().style.cursor = "";
    });

    map.on("click", "random-points-layer", e => {
      if (e.features.length) {
        const feature = e.features[0];
        const popupNode = document.createElement("div");
        ReactDOM.render(<Popup feature={feature} />, popupNode);
        popUpRef.current
          .setLngLat(feature.geometry.coordinates)
          .setDOMContent(popupNode)
          .addTo(map);
      }
    });

    return () => map.remove();
  }, []); 

  return <div className="map-container" ref={mapContainerRef} />;
};

export default MapComponent;
