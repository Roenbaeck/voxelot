I have a large documentation effort that needs to be done. 

Every configurable setting in config toml files needs to be documented in CONFIGURATION.md, section by section, using a nice consistent format. I want the default value to be shown in the documentation, along with a decription of what the setting controls, and the expected effect of changing the value. 

When you do this I want you to do the following things: 
1. Confirm that the setting is used in the code. Remove it from config.toml and config.rs if not.
2. Consider if the setting has a proper descriptive name. Change to a better one if not.
3. Consider if the setting is in the correct section. Move it if not.
4. Check if there are hard coded values close the where the setting is used that would also be good to have as settings.

You will also need to check and compare config.rs with the example config.toml, and ensure all settings have an entry in config.toml. 

Also check that section names use consistent terminology and change if you find inconsistencies.
