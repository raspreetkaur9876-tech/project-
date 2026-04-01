function checkSpam() {
    let text = document.getElementById("emailText").value;

    if (text === "") {
        document.getElementById("result").innerText = "Please enter email!";
        return;
    }

    if (text.includes("win") || text.includes("free")) {
        document.getElementById("result").innerText = "🚫 Spam";
    } else {
        document.getElementById("result").innerText = "✅ Not Spam";
    }
}