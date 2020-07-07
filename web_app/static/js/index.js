const GeneratorEndpoint = "generate"

document.addEventListener('DOMContentLoaded',  () => {
    const categories = document.querySelectorAll(".category-image")
    console.log(categories)
    categories.forEach(category => {
        category.addEventListener("click", () => {
            console.log(`Clicked on ${category.id}`)

            fetch(`${GeneratorEndpoint}/${category.id}`)
                .then(res => res.json())
                .then(data => {
                    console.log(data.result_image)
                    console.log(document.getElementById("result-image"))
                    document.getElementById("result-image").src = data.result_image
                })
        })
    })
})
