import Griddle, { plugins } from 'griddle-react';
import './Griddle.css'
const MapComponent = () => {
    var data=[
        {
            "Id":1,
          "District": "Adilabad",
          "Mandal": "Adilabad Rural",
          "Date": "2022-01-01",
          "Rain": 0,
          "Min_Temp": 14.1,
          "Max_Temp": 23.8,
          "Min_Humidity": 72.6,
          "Max_Humidity": 99.5,
          "Min_Wind_Speed": 0,
          "Max_Wind_Speed": 7.1,
          "latitude": 19.63629875,
          "longitude": 78.59593501523734
        },
        {
            "Id":2,
          "District": "Adilabad",
          "Mandal": "Adilabad Rural",
          "Date": "2022-01-02",
          "Rain": 0,
          "Min_Temp": 13.5,
          "Max_Temp": 27.4,
          "Min_Humidity": 55.6,
          "Max_Humidity": 99.8,
          "Min_Wind_Speed": 0,
          "Max_Wind_Speed": 8,
          "latitude": 19.63629875,
          "longitude": 78.59593501523734
        },
        {
            "Id":3,
          "District": "Adilabad",
          "Mandal": "Adilabad Rural",
          "Date": "2022-01-03",
          "Rain": 0,
          "Min_Temp": 12.9,
          "Max_Temp": 28.4,
          "Min_Humidity": 44.7,
          "Max_Humidity": 99,
          "Min_Wind_Speed": 0,
          "Max_Wind_Speed": 7.9,
          "latitude": 19.63629875,
          "longitude": 78.59593501523734
        },
        {
            "Id":4,
          "District": "Adilabad",
          "Mandal": "Adilabad Rural",
          "Date": "2022-01-04",
          "Rain": 0,
          "Min_Temp": 12.7,
          "Max_Temp": 27.6,
          "Min_Humidity": 49.5,
          "Max_Humidity": 99,
          "Min_Wind_Speed": 0,
          "Max_Wind_Speed": 9.7,
          "latitude": 19.63629875,
          "longitude": 78.59593501523734
        },
    ]      
    return           <Griddle
    data={data}
    components={{
       Filter: () => <span />,
      SettingsToggle: () => <span />,
    }}
    styleConfig={{
      icons: {
        TableHeadingCell: {
          sortDescendingIcon: '▼',
          sortAscendingIcon: '▲',
        },
      },
      classNames: {
        Cell: 'griddle-cell',
        Filter: 'griddle-filter',
        Loading: 'griddle-loadingResults',
        NextButton: 'griddle-next-button',
        NoResults: 'griddle-noResults',
        PageDropdown: 'griddle-page-select',
        Pagination: 'griddle-pagination',
        PreviousButton: 'griddle-previous-button',
        Row: 'griddle-row',
        RowDefinition: 'griddle-row-definition',
        Settings: 'griddle-settings',
        SettingsToggle: 'griddle-settings-toggle',
        Table: 'griddle-table',
        TableBody: 'griddle-table-body',
        TableHeading: 'griddle-table-heading',
        TableHeadingCell: 'griddle-table-heading-cell',
        TableHeadingCellAscending: 'griddle-heading-ascending',
        TableHeadingCellDescending: 'griddle-heading-descending',
      },
      styles: {},
    }}
  />

;
};

export default MapComponent;
