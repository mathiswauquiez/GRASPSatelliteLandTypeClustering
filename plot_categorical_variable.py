mapbox_access_token = "pk.eyJ1IjoibWF0aGlzdzU5IiwiYSI6ImNsaDZsYWs2czA3YWkzZnBlMnhtcmhyYW4ifQ.imLZJq1w2W6-yhuPQEb16Q"

def find_contours(im):
  contours, _ = cv2.findContours(im.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return contours

def im_ij_coordinates_to_lat_lon(i, j, shape = (180, 360)):

    # Get the lat and lon of the top left corner
    lat_tl, lon_tl = 90, -180

    # Get the lat and lon of the bottom right corner
    lat_br, lon_br = -90, 180

    # Get the lat and lon of the pixel
    lat = lat_tl + (lat_br - lat_tl) * (i / shape[0])
    lon = lon_tl + (lon_br - lon_tl) * (j / shape[1])

    return lon, lat



def im_contours_to_geojson(im, colormap = cm.viridis):
    to_lat_lon = lambda x : im_ij_coordinates_to_lat_lon(x[0], x[1], im.shape)
    id_value_map = []

    # Create the geojson
    geo = {"type": "FeatureCollection", "features": []}

    # Get the unique values
    unique_values = np.unique(im)

    for value in unique_values:
        if value == fill_value or np.isnan(value):
            continue
        # Create a binary image
        bin_im = (im == value).astype(np.uint8)
        # Find the contours
        contours = find_contours(bin_im)

        # Add the contours to the geojson
        for i, contour in enumerate(contours):
            if len(contour) < 2:
                continue

            """ We need to take into account the half pixel shift of the contours, for each pixel we need to have 4 coordinates"""
            contour = contour.squeeze()
            contour = np.concatenate([contour, contour[:1]])
            contour = np.flip(contour, axis=1)
            geo["features"].append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(map(to_lat_lon, contour.tolist()))], # we need to convert the contour to a list and the to the lat lon coordinates
                },
                "id" : i
            })
            id_value_map.append([i, im[contour[0][0], contour[0][1]]])



    return geo, pd.DataFrame({'id' : [x[0] for x in id_value_map], 'value' : [x[1] for x in id_value_map]})


def plot_categorical_variable(df, variable, colormap = cm.viridis):
    # Get the variable
    im = get_original_shape(df[variable])
    
    geo, targets = im_contours_to_geojson(im, colormap)




    fig = px.choropleth_mapbox(targets, geojson=geo,
                                locations='id', color='value',
                                color_continuous_scale="Viridis",
                                opacity=0.5,
                                labels={'value':'value'})

    fig.update_layout( # we update the layout
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            center=dict(
                lat=0,
                lon=0
            ),
            accesstoken=mapbox_access_token,
            bearing=0,
            pitch=0,
            zoom=1,
            style="satellite"
        ),
        width = 1500,
        height = 750
    )


    # Show the figure
    fig.show()

df['clusters'] = df['LandPercentage'] > 1
df['clusters'][df['clusters'] == False] = pd.NA
plot_categorical_variable(df, 'clusters')