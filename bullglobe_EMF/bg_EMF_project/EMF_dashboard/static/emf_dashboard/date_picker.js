
const datepicked = document.getElementById("date-picked").value;
console.log(datepicked)
const dashboardSplug = document.getElementById('dashboard-slug').textContent.trim();
console.log(dashboardSplug)
const socket = new WebSocket(`ws://${window.location.host}/ws/${dashboardSplug}/${datepicked}`);
console.log(socket)


submitBtn.addEventListener('click', ()=> {
    const dataValue = datepicked.value
    socket.send(JSON.stringify({
        'message': dataValue,
        // 'user': user,
    }));
})