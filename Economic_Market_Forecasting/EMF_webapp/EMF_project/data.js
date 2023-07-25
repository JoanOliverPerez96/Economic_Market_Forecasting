var endpint = '/api/chart/data/'
    $.ajax({
        method: 'GET',
        url: endpint,
        success: function(data){
            console.log(data)
        },
        error: function(error_data){
            console.log('error')
            console.log(error_data)
        }
    })