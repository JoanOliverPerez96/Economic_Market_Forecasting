console.log("hello world")
const dashboardSplug = document.getElementById('dashboard-slug').textContent.trim();
const user = document.getElementById('user').textContent.trim();
const submitBtn = document.getElementById('submit-btn');
const dataInput = document.getElementById('data-input').innerHTML.trim();
const dataBox = document.getElementById('data-box');


const socket = new WebSocket(`ws://${window.location.host}/ws/${dashboardSplug}/`);
console.log(socket)

socket.onmessage = function(e) {
    console.log('Server: ' + e.data);
    const {sender, message} = JSON.parse(e.data)
    console.log(message)
    dataBox.innerHTML += `<p>${sender}: ${message}</p>`
};

function get_date() {
    const datepicked = document.getElementById("data-input").value;
    console.log(datepicked)
    const dashboardSplug = document.getElementById('dashboard-slug').textContent.trim();
    console.log(dashboardSplug)
    const socket = new WebSocket(`ws://${window.location.host}/ws/${dashboardSplug}/?${datepicked}`);
    console.log(socket)
    return datepicked
}

// submitBtn.addEventListener('click', ()=> {
//     const dataValue = dataInput
//     socket.send(JSON.stringify({
//         'message': dataValue,
//         // 'user': user,
//     }));
// })

// socket.onopen = function(e) {
// };

// const fetchChartData = async () => {
//     const response = await fetch(window.location.href + 'chart/');
//     const data = await response.json();
//     console.log(data)
//     return data
// }
// fetchChartData()