// chess_scraper.js
// Description: This script scrapes chess games from the chess.com game archive page using Puppeteer and Cheerio.
// Author: Edward
// Date: 2024-08-16

const puppeteer = require('puppeteer');
const cheerio = require('cheerio');
const fs = require('fs');

(async () => {
    const browser = await puppeteer.launch({
        headless: true,
        args: ['--disable-javascript', '--disable-gpu', '--no-sandbox']
    });
    const page = await browser.newPage();

    const base_url = "https://www.chess.com/games/archive/EdwardL903?page=";
    const all_games = [];

    for (let page_num = 1; page_num <= 5; page_num++) { // Adjust range as needed for the number of pages
        const page_url = base_url + page_num;
        console.log(`Scraping page ${page_num}: ${page_url}`);
        
        await page.goto(page_url, { waitUntil: 'load', timeout: 0 });

        // Alternative to waitForTimeout
        await new Promise(r => setTimeout(r, 5000)); // Wait for 5 seconds

        const content = await page.content();
        const $ = cheerio.load(content);

        // Find the games table
        const games_table = $('table.archive-games-table');
        games_table.find('tr').each((index, element) => {
            const columns = $(element).find('td');
            if (columns.length > 1) { // Ensure it's a game row
                const game_data = {
                    var0: $(columns[0]).text().trim(),
                    var1: $(columns[1]).text().trim(),
                    var2: $(columns[2]).text().trim(),
                    var3: $(columns[3]).text().trim(),
                    var4: $(columns[4]).text().trim(),
                    var5: $(columns[5]).text().trim(),
                    var6: $(columns[6]).text().trim(),
                    // Add other fields as necessary
                };
                all_games.push(game_data);
            }
        });

        // Delay before moving to the next page to avoid hammering the server
        await new Promise(r => setTimeout(r, 5000)); // Wait for 5 seconds
    }

    // Save scraped data to a JSON file
    fs.writeFileSync('chess_games.json', JSON.stringify(all_games, null, 4));

    await browser.close();
    console.log(`Scraping completed. Total games scraped: ${all_games.length}`);
})();
