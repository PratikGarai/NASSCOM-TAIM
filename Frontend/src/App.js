import MapComponent from './MapComponent';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import DataComponent from "./DataComponent";
import NavBar from './NavBar';
function App() {
  return (
    <Router>
      <NavBar/>
      <div className="App">
        <div className="content">
          <Routes>
            <Route path="/" element={<MapComponent/>}/>
            <Route path="/Data" element={<DataComponent/>}/>
\          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
