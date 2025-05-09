# Dash application
#
# python dash_app.py

import os
import shutil
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import base64
import image_classifier_project


MEDIA_FOLDER = 'media'
EXTERNAL_STYLESHEETS = [
    'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
    {
        'href': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u',
        'crossorigin': 'anonymous'
    }
]

dict_extension_signature = {'png': '89504e470d0a', 'jpg': 'ffd8ff', 'jpeg': 'ffd8ff', 'bmp': '424d', 'gif': '47494638'
                            }


def remove_all_file_from_folder(folder_path):
    '''
    Remove all file and subfolder from a folder

    Args:
        folder_path (str): folder path

    Returns
        None
    '''
    print('Remove all file from media folder\n\t{}'.format(MEDIA_FOLDER))
    for file_object in os.listdir(folder_path):
        file_object_path = os.path.join(folder_path, file_object)
        if os.path.isfile(file_object_path) or os.path.islink(file_object_path):
            os.unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)


def save_image(name, content):
    '''
    Decode and store a file uploaded with Plotly Dash

    Args:
        name (str): image name
        content (str): image content

    Returns
        None
    '''
    image_filepath = os.path.join(MEDIA_FOLDER, name)
    data = content.encode('utf8').split(b';base64,')[1]
    with open(image_filepath, 'wb') as fp:
        print('Save uploaded image\n\t{}'.format(image_filepath))
        fp.write(base64.decodebytes(data))


def check_extension_match_signature(content_base64, extension, header_bytes=8):
    '''
    Check if the image extension matches the file signature

    Args:
        content (str): 
        extension (str): 
        header_bytes (int): 

    Returns:
        match (boolean): True if extension and signature mathces and False otherwise
    '''
    signature_hex = dict_extension_signature[extension]
    signature_byte = bytearray.fromhex(signature_hex)
    signature_base64 = base64.b64encode(signature_byte).decode()

    # print(signature_hex)
    # print(base64.b64decode(content_base64).hex()[0:len(signature_hex)])

    if signature_hex == base64.b64decode(content_base64).hex()[0:len(signature_hex)]:
        return True
    else:
        return False


def is_file_ok(name, content):
    '''
    Check the content of an image

    Args:
        name (str): image name
        content (str): image content

    Returns
        ok (boolean): True if the image is ok and False otherwise
    '''
    file_name, extension = name.rsplit('.', 1)
    if extension not in ['png', 'gif', 'jpg', 'jpeg', 'bmp']:
        return False
    else:
        file_type, rest = content.split('/', 1)
        image_type, rest = rest.split(';', 1)
        base, rest = rest.split(',', 1)
        # print(file_type)
        # print(image_type)
        # print(base)
        if file_type != 'data:image' or image_type not in ['png', 'gif', 'jpg', 'jpeg', 'bmp'] or base != 'base64':
            return False
        else:
            return check_extension_match_signature(rest, extension)


def get_encoded_image(image_filepath):
    '''
    Get image encoded using base64

    Args:
        image_filepath (str): image filepath

    Returns
        encoded image(str): encoded image in string format
    '''
    encoded_image = base64.b64encode(open(image_filepath, 'rb').read())
    file_name, extension = image_filepath.rsplit('.', 1)
    return 'data:image/{};base64,{}'.format(extension.lower(), encoded_image.decode())


def _create_app():
    ''' 
    Creates dash application

    Args:
        None

    Returns:
        app (dash.Dash): Dash application
    '''
    app = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)

    print('Check if media folder present and if not create it\n\t{}'.format(MEDIA_FOLDER))
    if not os.path.exists(MEDIA_FOLDER):
        os.makedirs(MEDIA_FOLDER)
    else:
        pass

    model, category_label_to_name = image_classifier_project.load_classifier()
    sample_training_dataset = image_classifier_project.get_sample_from_training_dataset(
        category_label_to_name)

    app.layout = dash.html.Div(
        [
            dbc.Nav(
                [
                    dash.html.Div(
                        [
                            dash.html.Div(
                                [
                                    dash.html.A('Image Classifier Application',
                                                href='/', className='navbar-brand')
                                ], className='navbar-header'), dash.html.Div(
                                [
                                    dash.html.Ul(
                                        [
                                            dash.html.Li(dash.html.A('Made with Udacity', href='https://www.udacity.com/')), dash.html.Li(
                                                dash.html.A('Github', href='https://github.com/simonerigoni/image_classifier_project'))
                                        ], className='nav navbar-nav')
                                ], className='collapse navbar-collapse')
                        ], className='container')
                ], className='navbar navbar-inverse navbar-fixed-top'), dash.html.Div(
                [
                    dash.html.H1('Image Classifier Application', className='text-center'), dash.html.P('Classify Images of Flowers', className='text-center'), dash.html.Hr(), dash.html.Div(
                        [
                            # dash.html.Div(
                            #    [
                            dash.dcc.Upload(id='upload-image', children=dash.html.Div(['Drag and Drop or ', dash.html.A('Select File')]),
                                            style={
                                            'width': '98%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                            },
                                            # Allow multiple files to be uploaded
                                            multiple=True
                                            ), dash.html.Div(id='output-image-upload'), dash.html.Hr(), dash.html.Button('Classify Image', id='button-submit', className='btn btn-lg btn-success')
                            # ] , className = 'row')
                        ], className='container')
                ], className='jumbotron'), dash.html.Div(id='results')
        ], className='container')

    def parse_contents(contents, filename, date):
        ''' 
        Parse loaded image

        Args:
            contents (str): image content in binary form
            filename (str): image filename
            date (str): loading image timestap 

        Returns:
            dash.html.div (dash_html_components.Div): div containing the input image
        '''
        return dash.html.Div([
            # dash.html.H5(filename)
            # , dash.html.H6(datetime.datetime.fromtimestamp(date))
            # HTML images accept base64 encoded strings in the same format that is supplied by the upload
            # ,
            dash.html.Img(src=contents, style={
                          'height': '75%', 'width': '75%'})
            # , dash.html.Hr()
            # , dash.html.Div('Raw Content')
            # , dash.html.Pre(contents[0:200] + '...', style = {'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'})
        ])

    @app.callback(dash.dependencies.Output('output-image-upload', 'children'), [dash.dependencies.Input('upload-image', 'contents')], [dash.dependencies.State('upload-image', 'filename'), dash.dependencies.State('upload-image', 'last_modified')])
    def update_output(list_of_contents, list_of_names, list_of_dates):
        '''
        Show loaded image

        Args:
            list_of_contents (list): list of uploaded images content
            list_of_names (list): list of uploaded images names
            list_of_dates (list): list of images loading timestap 

        Returns:
            children (dash_html_components.Div): html div to be updated
        '''
        if list_of_contents is not None:
            content = list_of_contents[0]
            name = list_of_names[0]
            date = list_of_dates[0]
            if is_file_ok(name, content):
                children = parse_contents(content, name, date)
                save_image(name, content)
            else:
                children = dash.html.Div(
                    'An error occurred: Accepted only png, gif, jpg, jpeg or bmp image')
            return children
        else:
            pass

    @app.callback(dash.dependencies.Output('results', 'children'), [dash.dependencies.Input('button-submit', 'n_clicks')])
    def update_results(n_click):
        '''
        Update the results section 

        Args:
            n_click (int): value of n_clicks of button-submit

        Returns:
            results (list): list of dash components 
        '''
        results = []
        files = os.listdir(MEDIA_FOLDER)
        number_of_classes = image_classifier_project.get_number_of_classes()

        if len(files) == 0:
            results.append(dash.html.Div(
                [
                    dash.html.H2('Overview of Training Dataset', className='text-center'), dash.html.H3(
                        '{} classes'.format(number_of_classes), className='text-center')
                ]))

            buffer_categories = []
            for category in list(sample_training_dataset.keys()):
                # results.append(dash.html.Div([dash.html.Img(src = get_encoded_image(sample_training_dataset[category]), style = {'height':'20%', 'width':'20%'}), dash.html.H5(category)]))
                if len(buffer_categories) == 4:
                    # print(buffer_categories[0], buffer_categories[1], buffer_categories[2], buffer_categories[3])
                    results.append(dash.html.Div([
                        dash.html.Div([dash.html.Img(src=get_encoded_image(sample_training_dataset[buffer_categories[0]]), style={'height': '75%', 'width': '75%'}), dash.html.H5(buffer_categories[0])], className='col-sm-3'), dash.html.Div([dash.html.Img(src=get_encoded_image(sample_training_dataset[buffer_categories[1]]), style={'height': '75%', 'width': '75%'}), dash.html.H5(buffer_categories[1])], className='col-sm-3'), dash.html.Div(
                            [dash.html.Img(src=get_encoded_image(sample_training_dataset[buffer_categories[2]]), style={'height': '75%', 'width': '75%'}), dash.html.H5(buffer_categories[2])], className='col-sm-3'), dash.html.Div([dash.html.Img(src=get_encoded_image(sample_training_dataset[buffer_categories[3]]), style={'height': '75%', 'width': '75%'}), dash.html.H5(buffer_categories[3])], className='col-sm-3')
                    ], className='row'))
                    buffer_categories = []
                else:
                    pass
                buffer_categories.append(category)
        else:
            image_filepath = MEDIA_FOLDER + '/' + files[0]
            print('Classify image\n\t{}'.format(image_filepath))
            category_probability = image_classifier_project.get_prediction(
                model, category_label_to_name, image_filepath)

            print(category_probability)

            results.append(dash.html.Div(
                [
                    dash.html.H2('Classification', className='text-center'), dash.dcc.Graph(
                        figure=go.Figure(
                            data=[
                                go.Bar(
                                    x=list(category_probability.keys()), y=list(category_probability.values()), name='Probability', marker=go.bar.Marker(color='rgb(55, 83, 109)')
                                )
                            ], layout=go.Layout(
                                title='Categories Probability Distribution', showlegend=False, legend=go.layout.Legend(x=0, y=1.0), margin=go.layout.Margin(l=40, r=0, t=40, b=30)
                            )
                        ), style={'height': 600}, id='categories-distribution-graph'), dash.html.Div(
                        [
                            dash.html.Div([dash.html.H3('Image to classify'), dash.html.Img(src=get_encoded_image(image_filepath), style={'height': '75%', 'width': '75%'})], className='col-sm-6'), dash.html.Div([dash.html.H3('Image from Training dataset for: {}'.format(
                                list(category_probability.keys())[0])), dash.html.Img(src=get_encoded_image(sample_training_dataset[list(category_probability.keys())[0]]), style={'height': '75%', 'width': '75%'})], className='col-sm-6')
                        ], className='row')
                ]))

            remove_all_file_from_folder(MEDIA_FOLDER)
        return results

    return app


if __name__ == '__main__':
    app = _create_app()
    app.run(debug=True)
else:
    pass
