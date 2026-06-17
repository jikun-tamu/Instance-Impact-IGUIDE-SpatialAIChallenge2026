# Building-Level Population Estimation System for Harris County, Texas

A comprehensive geospatial analysis system that estimates population at the individual building level using parcel data, building footprints, and US Census demographic data.

## 🎯 Overview

This system implements a sophisticated spatial disaggregation methodology to assign population estimates to individual buildings in Harris County, Texas. It combines:
- **Harris County parcel data** with land use classifications
- **Texas building footprint polygons** 
- **US Census ACS demographic data** at the Block Group level

The methodology produces building-level population estimates with quality flags and comprehensive validation metrics.

## 📊 Key Features

- **Spatial Accuracy**: Uses actual building footprints rather than statistical disaggregation
- **Local Context**: Block Group-level demographic ratios reflect local housing characteristics  
- **Quality Control**: Population caps and quality flags ensure realistic estimates
- **Scalable Performance**: Optimized for processing 100K+ buildings
- **Interactive Visualization**: Web-based maps with building-level detail
- **Comprehensive Validation**: Compares estimates to Census totals

## 🏗️ Methodology

### Core Population Assignment Formula
```
Estimated_Population = Estimated_Units × People_Per_Unit_Ratio × Occupancy_Rate
```

### Building Classification System
| Land Use Code | Building Type | Unit Estimation |
|---------------|---------------|-----------------|
| A1 | Single-family detached | 1 unit |
| A2 | Mobile home | 1 unit |
| B2 | Duplex | 2 units |
| B3 | Triplex | 3 units |
| B1, B4 | Multi-family | Area-based (800/650 sq ft per unit) |

### Data Sources
1. **Harris County Parcels**: Land use classifications (STAT_LAND_ field)
2. **Texas Building Footprints**: 10.6M+ building polygons statewide
3. **Census Block Groups**: TIGER/Line boundary files
4. **ACS 5-Year Data**: Tables B25032, B25033, B25002 for housing demographics

## 🚀 Quick Start

### Prerequisites
```bash
pip install geopandas pandas folium matplotlib seaborn requests shapely
```

Optional for enhanced functionality:
```bash
pip install pygris cenpy
```

### Basic Usage
1. **Load the Jupyter notebook**: `code.ipynb`
2. **Ensure data files are present**:
   - Harris County parcel shapefiles in `stratmap25-landparcels_48201_lp/shp/`
   - Texas building footprints as `Texas.geojson`
3. **Run cells sequentially** - the system will:
   - Load and process spatial data
   - Download Census demographics
   - Perform spatial joins and classification
   - Estimate building-level population
   - Generate validation metrics and visualizations

### Output Files
- `harris_county_building_population_estimates.csv` - Main results
- `harris_county_cbg_validation.csv` - Validation metrics
- `harris_county_buildings_with_population.geojson` - Spatial data for mapping
- `harris_county_population_map.html` - Interactive web map

## 🔬 Detailed Methodological Assessment

### 1. DATA SOURCES AND PREPROCESSING

#### 1.1 Core Datasets
- **Harris County Parcels**: 1,523,641 parcels with STAT_LAND_ classifications
- **Texas Building Footprints**: 10,678,921 statewide building polygons
- **Census Block Groups**: 2,830 CBGs for Harris County
- **ACS 5-Year Data**: 2018-2022 estimates for housing demographics

#### 1.2 Spatial Filtering Methodology
**Two-Stage Optimization Process**:
1. **Bounding Box Filter**: Rectangular bounds comparison eliminates ~90% of buildings
2. **Centroid-Based Filter**: Point-in-polygon test using building centroids vs. county boundary

**Key Assumption**: Building centroids adequately represent building location for geographic assignment.

### 2. BUILDING CLASSIFICATION AND UNIT ESTIMATION

#### 2.1 Land Use Code Mapping (STAT_LAND_ Field)
| Code | Classification | Units | Method |
|------|----------------|-------|--------|
| A1 | Single-family detached | 1 | Fixed |
| A2 | Mobile home | 1 | Fixed |
| B2 | Duplex | 2 | Fixed |
| B3 | Triplex | 3 | Fixed |
| B1, B4 | Multi-family | Variable | Area-based |

#### 2.2 Multi-Family Unit Estimation Thresholds
```python
# Critical Thresholds
AREA_THRESHOLD = 5,000 sq ft  # Distinguishes small vs large multi-family

# Small Multi-family (< 5,000 sq ft)
units = max(1, int(area_sqft / 800))  # 800 sq ft per unit

# Large Multi-family (≥ 5,000 sq ft)  
units = max(1, int(area_sqft / 650))  # 650 sq ft per unit
```

**Key Assumptions**:
- 800 sq ft/unit appropriate for small multi-family in Harris County
- 650 sq ft/unit appropriate for large multi-family complexes
- 5,000 sq ft threshold effectively distinguishes building types

#### 2.3 Mock Building Generation (When Real Data Unavailable)
**Building-to-Parcel Ratios**:
- A1, A2 (Single-family): 1 building per parcel
- B2 (Duplex): 1-2 buildings per parcel (random choice)
- B1, B3, B4 (Multi-family): 1-3 buildings per parcel (random)

**Building Size Ratios (% of parcel area)**:
- A1 (Single-family): 15-25% of parcel area
- A2 (Mobile home): 10-20% of parcel area  
- Multi-family: 20-40% of parcel area

**Spatial Placement**: Random position within 20-80% of parcel bounds (avoids edges)

### 3. CENSUS DEMOGRAPHIC DATA INTEGRATION

#### 3.1 Exact ACS Variables Used
**Population by Structure Type (Table B25033)**:
- B25033_002E: Population in 1-unit detached structures
- B25033_004E: Population in 2-unit structures
- B25033_005E: Population in 3-4 unit structures
- B25033_010E: Population in 5-9 unit structures
- B25033_011E: Population in 10+ unit structures

**Occupied Units by Structure Type (Table B25032)**:
- B25032_003E: Occupied 1-unit detached
- B25032_004E: Occupied 2-unit structures
- B25032_005E: Occupied 3-4 unit structures
- B25032_008E: Occupied 5-9 unit structures
- B25032_011E: Occupied 10+ unit structures

**Housing Occupancy (Table B25002)**:
- B25002_001E: Total housing units
- B25002_002E: Occupied housing units

#### 3.2 Ratio Calculation Methodology
**People Per Unit Ratios**:
```python
PPU_ratio = Population_in_structure_type / Occupied_units_structure_type
```

**Occupancy Rate**:
```python
Occupancy_rate = Occupied_housing_units / Total_housing_units
```

**Data Quality Controls**:
- **Bounds checking**: Ratios must be between 0.5 and 6.0 people/unit
- **Missing data handling**: Use defaults if numerator/denominator is 0 or null
- **API failure fallback**: Generate mock data with realistic distributions

#### 3.3 Structure Type Mapping to Building Classifications
- **Single-family detached** → B25033_002E / B25032_003E
- **Mobile home** → Same as single-family detached (assumption)
- **Duplex** → B25033_004E / B25032_004E
- **Triplex** → B25033_005E / B25032_005E  
- **Multi-family** → Weighted average: 60% of 5-9 unit ratio + 40% of 10+ unit ratio

**Key Assumption**: Multi-family weighting (60/40) reflects typical unit distribution in Harris County.

### 4. DEFAULT VALUES AND FALLBACK MECHANISMS

#### 4.1 Default People Per Unit Ratios (When ACS Data Missing)
```python
defaults = {
    'single_family_detached': 2.8,  # people/unit
    'mobile_home': 2.8,             # people/unit  
    'duplex': 2.5,                  # people/unit
    'triplex': 2.2,                 # people/unit
    'multi_family': 2.0             # people/unit
}
default_occupancy = 0.90  # 90% occupancy rate
```

#### 4.2 Actual Performance (Harris County Results)
**CBG-Specific vs Default Usage**:
- Single-family detached: 560/2830 CBGs used defaults (19.8%)
- Duplex: 0/2830 CBGs used defaults (0%)
- Triplex: 0/2830 CBGs used defaults (0%)
- Multi-family: 2/2830 CBGs used defaults (0.07%)
- Occupancy rate: 20/2830 CBGs used defaults (0.7%)

**Observed Ratios (Harris County)**:
- Single-family detached: mean=3.12, median=2.91
- Duplex: mean=2.49, median=2.50
- Triplex: mean=2.20, median=2.20
- Multi-family: mean=1.93, median=1.92
- Occupancy rate: mean=0.91, median=0.93

### 5. POPULATION ASSIGNMENT FORMULA

#### 5.1 Core Equation
```python
estimated_population = estimated_units × people_per_unit_ratio × occupancy_rate
```

#### 5.2 Population Caps (Maximum Reasonable Values)
```python
population_caps = {
    'single_family_detached': 12,  # people
    'mobile_home': 8,              # people
    'duplex': 20,                  # people (10 per unit)
    'triplex': 30,                 # people (10 per unit)
    'multi_family': estimated_units × 10  # 10 people per unit (dynamic)
}
```

#### 5.3 Quality Flag System
**Quality Categories**:
- **`ok`**: No issues detected
- **`minor_issues`**: Single issue (default PPU ratio OR default occupancy)
- **`major_issues`**: Multiple issues

**Quality Issue Types**:
- `default_ppu_ratio`: Used default people per unit ratio
- `default_occupancy`: Used default occupancy rate
- `population_capped`: Population exceeded caps and was reduced
- `zero_units`: Building has zero estimated units
- `very_large_building`: Building > 50,000 sq ft (may be institutional)
- `high_population_estimate`: Raw estimate > 100 people

### 6. COORDINATE SYSTEM AND SPATIAL OPERATIONS

#### 6.1 Coordinate Systems
- **Input Data**: EPSG:4326 (WGS84 Geographic)
- **Analysis**: EPSG:3081 (Texas State Plane Central, NAD83)
- **Output/Visualization**: EPSG:4326 (for web mapping)

**Rationale**: Texas State Plane provides accurate area calculations in feet for unit estimation.

#### 6.2 Spatial Join Methods
1. **Buildings to Parcels**: `within` predicate (building polygon within parcel polygon)
2. **Buildings to CBGs**: `within` predicate using building centroids (point-in-polygon)

**Key Assumption**: Building centroids adequately represent building location for CBG assignment.

### 7. VALIDATION METHODOLOGY

#### 7.1 Aggregate Validation
- **Method**: Sum building estimates by CBG, compare to ACS population totals
- **Metrics**: MAE, MAPE, R-squared correlation
- **Geographic Level**: Census Block Group (smallest available Census geography)

#### 7.2 Quality Assessment Metrics
- **Coverage**: % buildings successfully assigned to CBGs and parcels
- **Data Quality**: % using CBG-specific vs default demographic ratios
- **Estimation Issues**: Distribution of quality flags


## 🎨 Visualization Features

### Interactive Map (`harris_county_population_map.html`)
- **Building Polygons**: Color-coded by estimated population
- **Interactive Popups**: Click buildings for detailed information
- **Multiple Basemaps**: Street view, satellite imagery options
- **Performance Optimized**: Samples buildings for smooth interaction

### Static Visualizations
- Population distribution histograms
- Building type breakdowns  
- Validation scatter plots
- Geographic distribution maps

## 📊 Quality Assessment

### Quality Flags
- **`ok`**: No issues detected
- **`minor_issues`**: Used default demographic ratios
- **`major_issues`**: Multiple problems (capped population, missing data, etc.)

### Validation Metrics
- **Mean Absolute Error**: Average difference from Census totals
- **Mean Absolute Percentage Error**: Relative accuracy measure
- **R-squared**: Correlation with Census population data
- **Coverage Statistics**: Percentage of buildings successfully processed

## 🔧 Configuration Options

### Key Parameters
```python
# Test area size (for development/testing)
target_building_count = 3000

# Multi-family unit estimation thresholds
small_mf_threshold = 5000  # sq ft
small_mf_ratio = 800      # sq ft per unit
large_mf_ratio = 650      # sq ft per unit

# Population caps by building type
population_caps = {
    'single_family_detached': 12,
    'mobile_home': 8,
    'duplex': 20,
    'triplex': 30,
    'multi_family': 10  # per unit
}

# Default demographic ratios
defaults = {
    'people_per_unit_sfd': 2.8,
    'people_per_unit_mf': 2.0,
    'occupancy_rate': 0.90
}
```

### 8. KEY METHODOLOGICAL ASSUMPTIONS

#### 8.1 Spatial Assumptions
1. **Building centroids** adequately represent building location for CBG assignment
2. **"Within" spatial relationship** between buildings and parcels is appropriate
3. **Texas State Plane projection** provides sufficiently accurate area measurements

#### 8.2 Unit Estimation Assumptions
1. **800/650 sq ft ratios** are appropriate for Harris County multi-family housing
2. **5,000 sq ft threshold** effectively distinguishes small vs large multi-family
3. **Land use codes (STAT_LAND_)** accurately reflect actual building use

#### 8.3 Demographic Assumptions
1. **ACS 5-year estimates** represent current population characteristics
2. **Block Group-level ratios** are appropriate for individual buildings within CBG
3. **60/40 weighting** for multi-family ratios reflects local housing stock
4. **Population caps** reflect realistic household size distributions

#### 8.4 Mock Data Assumptions (When Real Buildings Unavailable)
1. **Building-to-parcel ratios** reflect typical development patterns
2. **Building size ratios** (15-40% of parcel area) are realistic
3. **Random spatial placement** within parcels is acceptable approximation

### 9. PERFORMANCE OPTIMIZATIONS

#### 9.1 Spatial Processing
- **Two-stage filtering**: Bounding box → centroid (10-50x speedup)
- **Coordinate system optimization**: Single transformation to Texas State Plane
- **Efficient boundary retrieval**: pygris vs expensive parcel union operations

#### 9.2 Data Processing
- **Chunked Census API calls**: 50 CBGs per request to avoid URL limits
- **Rate limiting**: 0.1 second delays between API calls
- **Smart sampling**: Limit visualizations to 25K buildings for performance

## 🚨 Known Limitations

### Data Dependencies
- **Building footprint quality**: Accuracy depends on source data currency and completeness
- **Parcel classification**: Assumes STAT_LAND_ codes accurately reflect building use
- **Census data lag**: ACS estimates have 5-year collection periods with inherent uncertainty

### Methodological Assumptions
- **Centroid assignment**: Building centroids adequately represent location for CBG assignment
- **Area-based estimation**: 800/650 sq ft ratios appropriate for local multi-family housing
- **Population caps**: Reflect realistic household size distributions for Harris County

### Temporal Considerations
- **Static analysis**: No temporal population dynamics (daily, seasonal variation)
- **Construction lag**: Building footprints may not reflect recent construction
- **Demographic change**: Population characteristics may have changed since ACS collection

## 🔄 Future Enhancements

### Potential Improvements
1. **Temporal Analysis**: Incorporate population change over time
2. **Building Age Integration**: Use construction dates to refine estimates
3. **Land Use Refinement**: Additional parcel characteristics for classification
4. **Validation Enhancement**: Ground-truth data for accuracy assessment
5. **Performance Scaling**: Distributed processing for state-wide analysis

### Additional Data Sources
- **Utility connections**: Validate occupancy assumptions
- **Property records**: Refine unit counts and building characteristics
- **Mobility data**: Validate population distribution patterns
- **Aerial imagery**: Automated building classification and change detection

## 📚 References

### Data Sources
- **Harris County Parcels**: StratMap Land Parcels Program
- **Texas Building Footprints**: Microsoft/Esri Building Footprints
- **Census Demographics**: US Census Bureau ACS 5-Year Estimates
- **Block Group Boundaries**: Census TIGER/Line Files

### Methodological References
- Spatial disaggregation techniques in population geography
- Census Bureau guidance on ACS data usage and limitations
- Best practices for building-level demographic estimation

## 📄 License

This project is intended for research and educational purposes. Please cite appropriately if used in academic work.

## 🤝 Contributing

Contributions welcome! Please focus on:
- Performance optimizations
- Additional validation methods
- Enhanced visualization capabilities
- Documentation improvements

## 📞 Contact

For questions about methodology or implementation, please refer to the detailed documentation in the Jupyter notebook.

---

**Note**: This system is designed for Harris County, Texas, but the methodology can be adapted for other jurisdictions with appropriate data sources and parameter adjustments.
