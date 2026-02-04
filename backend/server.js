const express = require("express");
const cors = require("cors");

const app = express();

app.use(cors());
app.use(express.json());

// routes
app.use("/api", require("./routes/testRoutes"));

const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Backend running on port ${PORT}`);
});
